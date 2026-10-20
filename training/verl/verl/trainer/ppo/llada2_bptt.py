"""Small, model-agnostic helpers for truncated LLaDA2 denoising BPTT."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class StagedBPTTReplay:
    """Recompute temporal VJPs in separate backwards so FSDP can reshard.

    The endpoint backward must finish before calling ``backward`` here. Its
    bridge gradient already includes loss normalization and AMP scaling; do
    not scale the predecessor VJPs again or step/zero the optimizer between
    stages. Every rank replays every slot, including zero-gradient padding.
    """

    def __init__(self, forward, *, profile_memory=False):
        self.forward = forward
        self.profile_memory = profile_memory
        self.records = []
        self.bridge = None

    def predecessor(self, inputs, chain_mask):
        device = inputs["input_ids"].device
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
        self.records.append((dict(inputs), chain_mask, cpu_rng, cuda_rng, device))
        with torch.no_grad():
            return self.forward(inputs)["bptt_self_conditioning_signal"]

    def endpoint_bridge(self, signal):
        self.bridge = signal.detach().requires_grad_(True)
        return self.bridge

    def backward(self):
        if self.bridge is None:
            raise RuntimeError("Staged BPTT has no endpoint bridge")
        gradient = self.bridge.grad
        if gradient is None:
            gradient = torch.zeros_like(self.bridge)
        self.bridge = None
        while self.records:
            slot = len(self.records) - 1
            inputs, chain_mask, cpu_rng, cuda_rng, device = self.records.pop()
            input_leaf = None
            if chain_mask is not None:
                input_leaf = inputs["_bptt_self_conditioning_signal"].detach().requires_grad_(True)
                inputs["_bptt_self_conditioning_signal"] = input_leaf
            # Reproduce stochastic operations without advancing the caller's
            # RNG a second time. Checkpoint recomputation runs in this context too.
            devices = [device.index] if device.type == "cuda" else []
            with torch.random.fork_rng(devices=devices):
                torch.set_rng_state(cpu_rng)
                if cuda_rng is not None:
                    torch.cuda.set_rng_state(cuda_rng, device)
                signal = self.forward(inputs)["bptt_self_conditioning_signal"]
                torch.autograd.backward(signal, gradient)
            if input_leaf is not None:
                gradient = input_leaf.grad
                if gradient is None:
                    gradient = torch.zeros_like(input_leaf)
                gradient = torch.where(chain_mask, gradient, 0)
            if self.profile_memory:
                print(
                    f"[BPTT_GPU] stage=predecessor_{slot}_backward "
                    f"allocated_gib={torch.cuda.memory_allocated() / 2**30:.3f} "
                    f"peak_gib={torch.cuda.max_memory_allocated() / 2**30:.3f}",
                    flush=True,
                )
            del signal, inputs, input_leaf


def _metadata(item: Mapping[str, Any]) -> Mapping[str, Any]:
    value = item.get("metadata")
    return value if isinstance(value, Mapping) else {}


def _state_block_and_step(
    step_key: int,
    item: Mapping[str, Any],
) -> tuple[int, int]:
    """Return producer block/local-step identity without doing key arithmetic."""

    metadata = _metadata(item)
    block = item.get(
        "block_idx",
        item.get(
            "block_index",
            item.get(
                "producer_block_index",
                metadata.get(
                    "block_idx",
                    metadata.get(
                        "block_index",
                        metadata.get("producer_block_index"),
                    ),
                ),
            ),
        ),
    )
    local_step = item.get("local_step", metadata.get("local_step"))
    if block is None or local_step is None:
        raise ValueError(
            "LLaDA2 BPTT requires block_idx and local_step on every replay state; "
            f"step_key={step_key} has block={block!r}, local_step={local_step!r}"
        )
    return int(block), int(local_step)


def plan_bptt_windows(
    states: Mapping[Any, Any],
    endpoint_step_keys: Iterable[int],
    *,
    horizon: int = 2,
) -> dict[int, tuple[int, ...]]:
    """Plan fixed-horizon windows ending at the already-selected states.

    Windows are ordered from oldest state to supervised endpoint. They are
    shortened at the beginning of a block and never cross block boundaries.
    """

    horizon = int(horizon)
    if horizon < 2:
        raise ValueError(f"BPTT horizon must be at least 2, got {horizon}")

    timelines: dict[int, list[tuple[int, int]]] = defaultdict(list)
    state_by_key: dict[int, Mapping[str, Any]] = {}
    for raw_key, raw_item in states.items():
        if not isinstance(raw_item, Mapping):
            continue
        step_key = int(raw_key)
        metadata = _metadata(raw_item)
        if bool(raw_item.get("is_terminal_commit", metadata.get("is_terminal_commit", False))):
            continue
        block, local_step = _state_block_and_step(step_key, raw_item)
        timelines[block].append((local_step, step_key))
        state_by_key[step_key] = raw_item

    positions: dict[int, tuple[list[tuple[int, int]], int]] = {}
    for timeline in timelines.values():
        timeline.sort(key=lambda pair: (pair[0], pair[1]))
        seen_local_steps: set[int] = set()
        for index, (local_step, step_key) in enumerate(timeline):
            if local_step in seen_local_steps:
                raise ValueError(
                    "LLaDA2 BPTT found duplicate local_step values in one block; "
                    f"local_step={local_step}, step_key={step_key}"
                )
            seen_local_steps.add(local_step)
            positions[step_key] = (timeline, index)

    windows: dict[int, tuple[int, ...]] = {}
    for raw_endpoint in endpoint_step_keys:
        endpoint = int(raw_endpoint)
        if endpoint not in state_by_key or endpoint not in positions:
            raise ValueError(
                f"selected BPTT endpoint {endpoint} is absent from captured replay states"
            )
        timeline, endpoint_index = positions[endpoint]
        start = max(0, endpoint_index - horizon + 1)
        window_pairs = timeline[start : endpoint_index + 1]
        for (previous_local, _), (next_local, _) in zip(
            window_pairs, window_pairs[1:]
        ):
            if next_local != previous_local + 1:
                raise ValueError(
                    "LLaDA2 BPTT window is missing an interior denoise state: "
                    f"endpoint={endpoint}, local_steps={previous_local}->{next_local}"
                )
        windows[endpoint] = tuple(step_key for _, step_key in window_pairs)
    return windows


def bptt_support_step_keys(windows: Mapping[int, Iterable[int]]) -> set[int]:
    """Return the replay rows that must be materialized for the planned windows."""

    return {int(step_key) for window in windows.values() for step_key in window}


def hidden_to_soft_embedding(
    hidden: torch.Tensor,
    *,
    lm_head: torch.nn.Module,
    embedding_weight: torch.Tensor,
    chunk_size: int = 8,
    checkpoint_projection: bool = True,
) -> torch.Tensor:
    """Compute the differentiable soft embedding with bounded saved activations.

    Chunking alone retains every chunk's vocabulary probabilities for backward.
    Non-reentrant checkpointing saves the inputs instead and recomputes those
    probabilities one chunk at a time, preserving gradients to H, W and E.
    """

    if hidden.dim() != 2:
        raise ValueError(f"hidden must be [tokens, hidden], got {tuple(hidden.shape)}")
    chunk_size = max(int(chunk_size), 1)
    def project(source, head_weight, head_bias, embeddings):
        logits = F.linear(source.to(dtype=head_weight.dtype), head_weight, head_bias)
        probabilities = F.softmax(logits.float(), dim=-1).to(dtype=embeddings.dtype)
        return probabilities.matmul(embeddings)

    chunks: list[torch.Tensor] = []
    for start in range(0, int(hidden.shape[0]), chunk_size):
        args = (
            hidden[start : start + chunk_size],
            lm_head.weight,
            getattr(lm_head, "bias", None),
            embedding_weight,
        )
        if checkpoint_projection and torch.is_grad_enabled():
            chunks.append(checkpoint(project, *args, use_reentrant=False))
        else:
            chunks.append(project(*args))
    if not chunks:
        return hidden.new_empty((0, int(embedding_weight.shape[1])))
    return torch.cat(chunks, dim=0)


def expand_bptt_block_signal(
    values: torch.Tensor,
    starts: torch.Tensor,
    lengths: torch.Tensor,
    *,
    response_length: int,
) -> torch.Tensor:
    """Restore one block per row, only after slicing a GPU microbatch."""
    if values.dim() == 3:
        values, starts, lengths = values[:, None], starts[:, None], lengths[:, None]
    if values.dim() != 4 or starts.shape != values.shape[:2] or lengths.shape != starts.shape:
        raise ValueError("BPTT blocks require [batch, pages, block, hidden] and [batch, pages] geometry")
    offsets = torch.arange(values.shape[2], device=values.device).view(1, 1, -1)
    positions = starts.unsqueeze(-1) + offsets
    valid = (offsets < lengths.unsqueeze(-1)) & (positions >= 0) & (positions < response_length)
    rows = torch.arange(values.shape[0], device=values.device).view(-1, 1, 1).expand_as(positions)
    result = values.new_zeros((values.shape[0], response_length, values.shape[3]))
    result[rows[valid], positions[valid]] = values[valid]
    return result


def bptt_token_validity(valid, starts, lengths, *, block_size, response_length):
    """Expand temporal validity per page, without letting padding enable a chain."""
    values = valid[:, :, None, None].expand(-1, -1, block_size, 1)
    return expand_bptt_block_signal(values, starts, lengths, response_length=response_length)


def pack_bptt_replay_rows(tensors, payload, sample_indices, step_keys, *, response_length,
                          block_size, max_blocks=32):
    """Pack disjoint noisy blocks of one response, keeping every temporal edge.

    Clean context is shared; each noisy block remains isolated by the existing
    preexpanded block attention mask. Fields stay row-aligned for TensorDict
    slicing. Hidden pages are expanded only inside the actor microbatch.
    Returns endpoint tensors, temporal payload, and original row indices/group.
    """
    if max_blocks < 1:
        raise ValueError("BPTT packing_max_blocks must be positive")
    starts = payload["rollout_self_conditioning_signal_block_starts"].tolist()
    lengths = payload["rollout_self_conditioning_signal_block_lengths"].tolist()
    by_sample = defaultdict(lambda: defaultdict(list))
    singleton_rows = []
    for row, (sample, start, length) in enumerate(zip(sample_indices, starts, lengths, strict=True)):
        if start < 0 or length <= 0 or start + length > response_length:
            raise ValueError("BPTT packing requires bounded, nonempty block spans")
        if start // block_size != (start + length - 1) // block_size:
            raise ValueError("BPTT replay span crosses a physical block boundary")
        # Captured/normalized predecessor pages can have a different span from
        # the endpoint (e.g. a clipped prompt/tail page). Such windows still
        # have valid unpacked replay semantics, but cannot share this block's
        # packing geometry. Preserve their complete original rows.
        if not bool((payload["bptt_block_starts"][row] == start).all()) or not bool(
                (payload["bptt_block_lengths"][row] == length).all()):
            singleton_rows.append(row)
            continue
        by_sample[int(sample)][start // block_size].append(row)
    groups = []
    for blocks in by_sample.values():
        timelines = [sorted(rows, key=lambda r: step_keys[r]) for _, rows in sorted(blocks.items())]
        for local_round in range(max(map(len, timelines))):
            rows = [timeline[local_round] for timeline in timelines if local_round < len(timeline)]
            groups.extend(rows[i:i + max_blocks] for i in range(0, len(rows), max_blocks))
    groups.extend([row] for row in singleton_rows)
    if not groups:
        raise ValueError("Cannot pack an empty BPTT batch")
    device = tensors["input_ids"].device
    representatives = torch.tensor([group[0] for group in groups], device=device)
    width = max(map(len, groups))
    index = torch.zeros((len(groups), width), dtype=torch.long, device=device)
    present = torch.zeros_like(index, dtype=torch.bool)
    for row, group in enumerate(groups):
        index[row, :len(group)] = torch.tensor(group, device=device)
        present[row, :len(group)] = True
    packed = {key: value.index_select(0, representatives) if value is not None else None
              for key, value in tensors.items()}
    prefix = tensors["input_ids"].shape[-1] - response_length
    for key in ("labels", "dense_labels"):
        if packed.get(key) is not None:
            packed[key][:, -response_length:] = -100
    for key, fill in (("step_maps", -1), ("pmask", False), ("ge_mask", False)):
        packed[key].fill_(fill)
    packed_payload = {}
    for name in ("rollout_self_conditioning_signal", "rollout_logits_source_hidden"):
        packed_payload[name + "_block_values"] = payload[name + "_block_values"][index]
        packed_payload[name + "_block_starts"] = payload[name + "_block_starts"][index]
        packed_payload[name + "_block_lengths"] = payload[name + "_block_lengths"][index] * present
    for name in ("bptt_block_starts", "bptt_block_lengths", "bptt_step_keys", "bptt_valid_mask",
                 "bptt_rollout_self_conditioning_signal"):
        packed_payload[name] = payload[name][index].movedim(2, 1)
    packed_payload["bptt_block_lengths"] *= present[:, None]
    packed_payload["bptt_valid_mask"] &= present[:, None]
    packed_payload["bptt_step_keys"].masked_fill_(~present[:, None], -1)
    packed_payload["bptt_packed_singleton"] = torch.tensor(
        [len(group) == 1 for group in groups], dtype=torch.bool, device=device)
    packed_payload["bptt_input_ids"] = payload["bptt_input_ids"][representatives]
    packed_payload["bptt_responses"] = payload["bptt_responses"][representatives]
    for target, group in enumerate(groups):
        if len(group) > 1:
            packed_payload["bptt_responses"][target].fill_(-100)
        for source in group:
            start, length = starts[source], lengths[source]
            span = slice(start, start + length)
            absolute = slice(prefix + start, prefix + start + length)
            if not torch.equal(tensors["input_ids"][source, :prefix], packed["input_ids"][target, :prefix]):
                raise ValueError("BPTT packed rows must share identical clean context")
            if not torch.equal(payload["bptt_input_ids"][source, :, :prefix],
                               packed_payload["bptt_input_ids"][target, :, :prefix]):
                raise ValueError("BPTT packed predecessors must share identical clean context")
            for name in ("input_ids", "labels", "dense_labels"):
                if packed.get(name) is not None:
                    packed[name][target, absolute] = tensors[name][source, absolute]
            for name in ("step_maps", "pmask", "ge_mask"):
                packed[name][target, span] = tensors[name][source, span]
            if len(group) > 1:
                packed_payload["bptt_input_ids"][target, :, absolute] = payload["bptt_input_ids"][source, :, absolute]
                packed_payload["bptt_responses"][target, :, span] = payload["bptt_responses"][source, :, span]
    if singleton_rows:
        examples = [dict(row=row, endpoint=(starts[row], lengths[row]),
                         predecessor_starts=payload["bptt_block_starts"][row].tolist(),
                         predecessor_lengths=payload["bptt_block_lengths"][row].tolist())
                    for row in singleton_rows[:3]]
        print(f"[BPTT_PACKING] variable_span_singletons={len(singleton_rows)} examples={examples}", flush=True)
    if int(packed["labels"].ne(-100).sum()) != int(tensors["labels"].ne(-100).sum()):
        raise RuntimeError("BPTT packing changed the supervised token count")
    return packed, packed_payload, groups
