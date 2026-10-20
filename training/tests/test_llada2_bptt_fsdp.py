"""Two-rank CPU FSDP regression for the real actor BPTT orchestration.

Run directly in the training container; no GPUs or full model are needed.
Checks gradients across accumulated updates and full-weight release at the
temporal boundary, including the actor's projection outside the root forward.
"""

import functools
import tempfile
import types
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.checkpoint import checkpoint

from test_llada2_bptt import CompactBPTTTest, hidden_to_soft_embedding


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)

    def forward(self, x):
        return self.linear(x).tanh()


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(17, 8)
        self.head = nn.Linear(8, 17, bias=False)
        self.adapter = nn.Linear(8, 8)
        self.blocks = nn.ModuleList([Block() for _ in range(4)])

    def forward(self, ids, signal):
        signal = torch.nn.functional.pad(signal, (0, 0, ids.shape[-1] - signal.shape[1], 0))
        hidden = self.embedding(ids) + self.adapter(signal)
        for block in self.blocks:
            hidden = checkpoint(block, hidden, use_reentrant=False)
        return hidden


def worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank, world_size=2)
    policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})
    reference = None
    for staged in (False, True):
        torch.manual_seed(3)
        model = FSDP(Model(), auto_wrap_policy=policy, device_id=torch.device("cpu"))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        handles = [m._handle for m in FSDP.fsdp_modules(model)
                   if isinstance(m._fsdp_wrapped_module, Block)]
        retained = []
        def observe(grad):
            retained.append(sum(h.flat_param._full_param_padded.untyped_storage().nbytes() > 0
                                for h in handles))
            return grad
        def forward(data, **kwargs):
            signal = data.get("_bptt_self_conditioning_signal", data["rollout_self_conditioning_signal"])
            if not data.get("_bptt_return_signal_only"):
                signal.register_hook(observe)
            hidden = model(data["input_ids"], signal)
            if data.get("_bptt_return_signal_only"):
                value = hidden_to_soft_embedding(hidden.flatten(0, 1), lm_head=model.module.head,
                    embedding_weight=model.module.embedding.weight, chunk_size=2)
                return {"bptt_self_conditioning_signal": value.view_as(hidden)}
            return {"loss": model.module.head(hidden).square().mean(), "generation_jsd_metrics": {}}
        actor = types.SimpleNamespace(_forward_micro_batch=forward, config={})
        snapshots = []
        for update in range(2):
            optimizer.zero_grad(set_to_none=True)
            for depth in (1, 2):
                torch.manual_seed(31 + rank + depth + update)
                valid = torch.ones(2, depth, dtype=torch.bool)
                valid[0, :] = False
                # Opposite padding patterns on ranks: execution must stay aligned.
                if rank == 1:
                    valid[1, :-1] = False
                inputs = {
                    "input_ids": torch.zeros(2, 3, dtype=torch.long),
                    "responses": torch.ones(2, 3, dtype=torch.long),
                    "bptt_input_ids": torch.zeros(2, depth, 3, dtype=torch.long),
                    "bptt_responses": torch.ones(2, depth, 3, dtype=torch.long),
                    "bptt_valid_mask": valid,
                    "rollout_self_conditioning_signal": torch.randn(2, 3, 8),
                    "bptt_rollout_self_conditioning_signal": torch.randn(2, depth, 3, 8),
                }
                result = CompactBPTTTest.actor_method()(actor, inputs, temperature=1, staged_backward=staged)
                (result["loss"] * 0.3).backward()
                if staged:
                    result.pop("_bptt_backward_replay").backward()
                snapshots.append([p.grad.detach().clone() for p in model.parameters()])
            optimizer.step()
        if staged:
            for actual, expected in zip(snapshots, reference):
                for a, b in zip(actual, expected):
                    torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
            assert max(retained) <= 1, retained
        else:
            reference = snapshots
            assert min(retained) == len(handles), retained
        if rank == 0:
            print(f"staged={staged}: full decoder weights retained at bridge={retained}", flush=True)
        del actor, optimizer, model
        dist.barrier()
    if rank == 0:
        print("FSDP gradient parity and staged weight release passed across two accumulated updates", flush=True)
    # The same production orchestration with mixed temporal validity per page.
    from test_llada2_bptt_packing import pack_fixture
    original, packed, _ = pack_fixture(depth=2, variable_spans=True)
    reference = None
    for tensors, payload, samples in (original, packed):
        torch.manual_seed(7)
        model = FSDP(Model(), auto_wrap_policy=policy, device_id=torch.device("cpu"))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        def packed_forward(data, **kwargs):
            signal = data.get("_bptt_self_conditioning_signal", data["rollout_self_conditioning_signal"])
            hidden = model(data["input_ids"], signal)[:, -8:]
            valid = data["responses"].ne(-100)
            if data.get("_bptt_return_signal_only"):
                values = hidden_to_soft_embedding(hidden[valid], lm_head=model.module.head,
                    embedding_weight=model.module.embedding.weight, chunk_size=2)
                return {"bptt_self_conditioning_signal": torch.zeros_like(hidden).masked_scatter(
                    valid[:, :, None], values)}
            logits = model.module.head(hidden)
            loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), data["responses"].flatten(),
                ignore_index=-100, reduction="sum") * (rank + 1) / 48
            return {"loss": loss, "generation_jsd_metrics": {}}
        actor = types.SimpleNamespace(_forward_micro_batch=packed_forward, config={})
        snapshots = []
        for update in range(2):
            optimizer.zero_grad(set_to_none=True)
            for row in range(len(samples)):
                inputs = {k: v[row:row+1] for k, v in payload.items()}
                inputs["input_ids"] = tensors["input_ids"][row:row+1]
                inputs["responses"] = tensors["labels"][row:row+1, -8:]
                result = CompactBPTTTest.actor_method()(actor, inputs, temperature=1, staged_backward=True)
                result["loss"].backward()
                result.pop("_bptt_backward_replay").backward()
            snapshots.append([p.grad.detach().clone() for p in model.parameters()])
            optimizer.step()
        if reference is None:
            reference = snapshots
        else:
            for actual, expected in zip(snapshots, reference):
                for a, b in zip(actual, expected):
                    torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
        del actor, optimizer, model
        dist.barrier()
    if rank == 0:
        print("Packed/unpacked FSDP gradients match across two accumulated optimizer updates", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="orl-bptt-fsdp-") as directory:
        mp.spawn(worker, args=(str(Path(directory) / "rendezvous"),), nprocs=2, join=True)
