"""Packing parity through block attention, temporal VJPs, CE and entropy loss."""
import ast
import types
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from test_llada2_bptt import CompactBPTTTest, _MODULE


def replay_fixture(depth=2, variable_spans=False):
    torch.manual_seed(83)
    response, block, hidden, prefix = 8, 3, 8, 10
    rows = [(sample, b, step) for sample in range(2)
            for b, count in enumerate((4, 2, 3)) for step in range(count)]
    n = len(rows)
    clean = torch.randint(0, 17, (2, prefix))
    ids = torch.stack([torch.cat([clean[s], clean[s, -response:]]) for s, _, _ in rows])
    labels = torch.full_like(ids, -100)
    maps = torch.full((n, response), -1, dtype=torch.long)
    pred_ids = ids[:, None].expand(-1, depth, -1).clone()
    pred_labels = torch.full((n, depth, response), -100, dtype=torch.long)
    starts = torch.tensor([b * block for _, b, _ in rows])
    lengths = torch.tensor([min(block, response - int(start)) for start in starts])
    valid = torch.zeros(n, depth, dtype=torch.bool)
    keys = [b * 100 + step for _, b, step in rows]
    for row, (sample, b, step) in enumerate(rows):
        start, length = int(starts[row]), int(lengths[row])
        span = slice(start, start + length)
        ids[row, prefix + start:prefix + start + length] = (step + b) % 17
        labels[row, prefix + start:prefix + start + length] = clean[sample, 2 + start:2 + start + length]
        maps[row, span] = keys[row]
        pred_labels[row, :, span] = labels[row, prefix + start:prefix + start + length]
        for slot in range(depth):
            pred_step = step - depth + slot
            valid[row, slot] = pred_step >= 0
            pred_ids[row, slot, prefix + start:prefix + start + length] = max(pred_step + b, 0) % 17
    payload = {
        "bptt_input_ids": pred_ids, "bptt_responses": pred_labels,
        "bptt_block_starts": starts[:, None].expand(-1, depth).clone(),
        "bptt_block_lengths": lengths[:, None].expand(-1, depth).clone(),
        "bptt_step_keys": torch.arange(n * depth).reshape(n, depth),
        "bptt_valid_mask": valid,
        "bptt_rollout_self_conditioning_signal": torch.randn(n, depth, block, hidden),
    }
    for field in ("rollout_self_conditioning_signal", "rollout_logits_source_hidden"):
        payload[field + "_block_values"] = torch.randn(n, block, hidden)
        payload[field + "_block_starts"] = starts.clone()
        payload[field + "_block_lengths"] = lengths.clone()
    tensors = dict(input_ids=ids, labels=labels, dense_labels=None, step_maps=maps,
                   pmask=maps.ge(0), ge_mask=maps.ge(0), lp=torch.zeros(n, response))
    if variable_spans:
        # Real captures may mix full and normalized/clipped spans within a
        # producer block's timeline. Include changed length, offset and bucket.
        for row, start, length in ((3, 0, 2), (5, 4, 2), (8, 3, 3)):
            payload["bptt_block_starts"][row, -1] = start
            payload["bptt_block_lengths"][row, -1] = length
            payload["bptt_responses"][row, -1].fill_(-100)
            payload["bptt_responses"][row, -1, start:start+length] = clean[0, 2+start:2+start+length]
            payload["bptt_input_ids"][row, -1, prefix+start:prefix+start+length] = 16
    return tensors, payload, [r[0] for r in rows], keys


def pack_fixture(depth=2, max_blocks=32, variable_spans=False):
    tensors, payload, samples, keys = replay_fixture(depth, variable_spans=variable_spans)
    packed, temporal, groups = _MODULE.pack_bptt_replay_rows(
        tensors, payload, samples, keys, response_length=8, block_size=3, max_blocks=max_blocks)
    return (tensors, payload, samples), (packed, temporal, [samples[g[0]] for g in groups]), groups


def actual_visibility_mask():
    path = Path(__file__).parents[1] / "overlays/llada2/modeling_llada2_moe.py"
    tree = ast.parse(path.read_text())
    method = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "_preexpanded_visibility_mask")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[method.name]({"cu_seqlens": torch.tensor([0, 18]),
        "L0": torch.tensor([2]), "L1": torch.tensor([8]), "staircase_size": 3}, 18, "cpu")


class TinyAttentionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(17, 8)
        self.adapter = torch.nn.Linear(8, 8, bias=False)
        self.layers = torch.nn.ModuleList([torch.nn.Linear(8, 24) for _ in range(2)])
        self.head = torch.nn.Linear(8, 17, bias=False)
        self.register_buffer("mask", actual_visibility_mask())

    def forward(self, ids, signal):
        x = self.embedding(ids) + self.adapter(F.pad(signal.to(self.embedding.weight.dtype), (0, 0, 10, 0)))
        for layer in self.layers:
            q, k, v = layer(x).chunk(3, dim=-1)
            scores = (q @ k.transpose(-1, -2) / 8**0.5).masked_fill(~self.mask, -torch.inf)
            x = (x + scores.softmax(-1) @ v).tanh()
        return x[:, -8:]


class PackingTest(unittest.TestCase):
    def test_geometry_coverage_padding_and_slice_alignment(self):
        for cap in (1, 2, 32):
            original, packed, groups = pack_fixture(max_blocks=cap)
            tensors, payload, samples = original
            actual, temporal, packed_samples = packed
            self.assertEqual(sorted(i for group in groups for i in group), list(range(len(samples))))
            self.assertLessEqual(max(map(len, groups)), cap)
            self.assertEqual(int(actual["labels"].ne(-100).sum()), int(tensors["labels"].ne(-100).sum()))
            self.assertEqual(int(temporal["bptt_valid_mask"].sum()), int(payload["bptt_valid_mask"].sum()))
            for row, group in enumerate(groups):
                self.assertEqual(len({samples[i] for i in group}), 1)
                self.assertEqual(len({int(payload["rollout_self_conditioning_signal_block_starts"][i]) // 3
                                      for i in group}), len(group))
                # This is also the actor's expansion after slicing one GPU microbatch.
                dense = _MODULE.expand_bptt_block_signal(
                    temporal["rollout_self_conditioning_signal_block_values"][row:row+1],
                    temporal["rollout_self_conditioning_signal_block_starts"][row:row+1],
                    temporal["rollout_self_conditioning_signal_block_lengths"][row:row+1], response_length=8)
                for source in group:
                    start = int(payload["rollout_self_conditioning_signal_block_starts"][source])
                    length = int(payload["rollout_self_conditioning_signal_block_lengths"][source])
                    torch.testing.assert_close(dense[0, start:start+length],
                        payload["rollout_self_conditioning_signal_block_values"][source, :length])
                    torch.testing.assert_close(actual["step_maps"][row, start:start+length],
                                               tensors["step_maps"][source, start:start+length])

    def test_attention_loss_temporal_gradient_and_update_parity(self):
        for depth, variable_spans in ((1, False), (2, False), (1, True), (2, True)):
            original, packed, _ = pack_fixture(depth, variable_spans=variable_spans)
            results = []
            for data, staged in ((original, False), (original, True), (packed, False), (packed, True)):
                torch.manual_seed(29)
                # Float64 avoids Adam amplifying roundoff in the theoretically
                # zero attention-key bias gradient into a spurious update gap.
                model = TinyAttentionModel().double()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                tensors, payload, samples = data
                def forward(batch, **kwargs):
                    signal = batch.get("_bptt_self_conditioning_signal", batch["rollout_self_conditioning_signal"])
                    hidden = model(batch["input_ids"], signal)
                    valid = batch["responses"].ne(-100)
                    if batch.get("_bptt_return_signal_only"):
                        values = _MODULE.hidden_to_soft_embedding(hidden[valid], lm_head=model.head,
                            embedding_weight=model.embedding.weight, chunk_size=2)
                        output = torch.zeros_like(hidden).masked_scatter(valid[:, :, None], values)
                        return {"bptt_self_conditioning_signal": output}
                    logits = model.head(hidden)
                    ce = F.cross_entropy(logits.flatten(0, 1), batch["responses"].flatten(),
                                         ignore_index=-100, reduction="sum")
                    # One entropy hinge per original state/page, not per packed row.
                    entropy = -(logits.log_softmax(-1) * logits.softmax(-1)).sum(-1)
                    aec = logits.sum() * 0
                    for state in batch["step_maps"].unique():
                        if state >= 0:
                            mask = batch["step_maps"].eq(state) & valid
                            aec = aec + F.relu(entropy[mask].mean() - 0.2)
                    return {"loss": (ce + 0.4 * aec) / 48, "generation_jsd_metrics": {}}
                actor = types.SimpleNamespace(_forward_micro_batch=forward, config={})
                losses = []
                for row in range(len(samples)):
                    batch = {k: v[row:row+1] for k, v in payload.items()}
                    batch.update({k: v[row:row+1] for k, v in tensors.items() if v is not None})
                    batch["responses"] = batch.pop("labels")[:, -8:]
                    result = CompactBPTTTest.actor_method()(actor, batch, temperature=1, staged_backward=staged)
                    losses.append(result["loss"].detach())
                    result["loss"].backward()
                    if staged:
                        result.pop("_bptt_backward_replay").backward()
                grads = [p.grad.clone() for p in model.parameters()]
                optimizer.step()
                results.append((sum(losses), grads, [p.detach().clone() for p in model.parameters()]))
            for actual in results[1:]:
                torch.testing.assert_close(actual[0], results[0][0], atol=2e-6, rtol=2e-5)
                for a, b in zip(actual[1] + actual[2], results[0][1] + results[0][2]):
                    torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)

    def test_variable_span_windows_preserve_complete_predecessors(self):
        original, packed, groups = pack_fixture(variable_spans=True)
        _, payload, samples = original
        _, temporal, _ = packed
        self.assertLess(len(groups), len(samples))
        self.assertEqual(sorted(i for group in groups for i in group), list(range(len(samples))))
        self.assertEqual(int(temporal["bptt_valid_mask"].sum()), int(payload["bptt_valid_mask"].sum()))
        for source in (3, 5, 8):
            target = groups.index([source])
            self.assertTrue(temporal["bptt_packed_singleton"][target])
            for key in ("bptt_input_ids", "bptt_responses"):
                torch.testing.assert_close(temporal[key][target], payload[key][source])
            for key in ("bptt_block_starts", "bptt_block_lengths", "bptt_valid_mask"):
                torch.testing.assert_close(temporal[key][target, :, 0], payload[key][source])

    def test_mismatched_context_is_rejected(self):
        tensors, payload, samples, keys = replay_fixture()
        tensors["input_ids"][4, 0] += 1
        with self.assertRaisesRegex(ValueError, "clean context"):
            _MODULE.pack_bptt_replay_rows(tensors, payload, samples, keys,
                response_length=8, block_size=3)

    def test_actual_aec_builder_preserves_pairs_and_statistics(self):
        import numpy as np
        from typing import Any, Optional
        path = Path(__file__).parents[1] / "verl/verl/trainer/ppo/ray_trainer.py"
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RayPPOTrainer")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                      and n.name == "_build_adjacent_entropy_actor_tensors")
        namespace = dict(torch=torch, np=np, Any=Any, Optional=Optional)
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
        original, packed, groups = pack_fixture()
        states = [{}, {}]
        tensors, payload, samples = original
        for row, sample in enumerate(samples):
            key = int(tensors["step_maps"][row].max())
            length = int(payload["rollout_self_conditioning_signal_block_lengths"][row])
            states[sample][key] = dict(adjacent_entropy_pair_valid=True,
                adjacent_entropy_next_mask=torch.ones(length, dtype=torch.bool),
                adjacent_entropy_prev_mean=0.3 + row * 0.01,
                adjacent_entropy_pair_local_step=key % 100,
                adjacent_entropy_rollout_next_mean=0.2, adjacent_entropy_vocab_size=17)
        results = []
        for t, _, source_ids in (original, packed):
            sc = np.empty(len(source_ids), dtype=object)
            sc[:] = [states[s] for s in source_ids]
            results.append(namespace[method.name](None, step_maps=t["step_maps"],
                rollout_self_conditioning=sc, response_len=8))
        for target, group in enumerate(groups):
            for source in group:
                mask = tensors["step_maps"][source].ge(0)
                for key in ("adjacent_entropy_pair_mask", "adjacent_entropy_prev_mean",
                            "adjacent_entropy_pair_local_step", "adjacent_entropy_rollout_next_mean",
                            "adjacent_entropy_vocab_size"):
                    torch.testing.assert_close(results[0][key][source, mask], results[1][key][target, mask])
        for result in results:
            ids = result["adjacent_entropy_pair_ids"]
            self.assertEqual(ids[ids.ge(0)].unique().numel(), len(samples))


if __name__ == "__main__":
    unittest.main()
