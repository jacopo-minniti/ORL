import ast
import types
import importlib.util
from pathlib import Path
import unittest

import torch


_MODULE_PATH = (
    Path(__file__).parents[1]
    / "verl"
    / "verl"
    / "trainer"
    / "ppo"
    / "llada2_bptt.py"
)
_SPEC = importlib.util.spec_from_file_location("llada2_bptt", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
bptt_support_step_keys = _MODULE.bptt_support_step_keys
hidden_to_soft_embedding = _MODULE.hidden_to_soft_embedding
plan_bptt_windows = _MODULE.plan_bptt_windows


def _state(block, local_step):
    return {"block_idx": block, "local_step": local_step}


class LLaDA2BPTTTest(unittest.TestCase):
    def test_windows_add_only_same_block_predecessors(self):
        states = {
            100: _state(0, 1),
            101: _state(0, 2),
            102: _state(0, 3),
            200: _state(1, 1),
            201: _state(1, 2),
        }

        windows = plan_bptt_windows(states, {102, 200, 201}, horizon=2)

        self.assertEqual(
            windows,
            {102: (101, 102), 200: (200,), 201: (200, 201)},
        )
        self.assertEqual(
            bptt_support_step_keys(windows), {101, 102, 200, 201}
        )

    def test_horizon_is_truncated_at_block_start(self):
        states = {10 + i: _state(3, i + 1) for i in range(4)}
        self.assertEqual(
            plan_bptt_windows(states, {10, 12, 13}, horizon=3),
            {
                10: (10,),
                12: (10, 11, 12),
                13: (11, 12, 13),
            },
        )

    def test_soft_embedding_keeps_gradient_bridge(self):
        torch.manual_seed(0)
        hidden = torch.randn(3, 5, requires_grad=True)
        lm_head = torch.nn.Linear(5, 7, bias=False)
        embedding = torch.randn(7, 5, requires_grad=True)

        signal = hidden_to_soft_embedding(
            hidden,
            lm_head=lm_head,
            embedding_weight=embedding,
            chunk_size=2,
        )
        signal.square().sum().backward()

        self.assertIsNotNone(hidden.grad)
        self.assertGreater(torch.count_nonzero(hidden.grad), 0)
        self.assertIsNotNone(lm_head.weight.grad)
        self.assertGreater(torch.count_nonzero(lm_head.weight.grad), 0)
        self.assertIsNotNone(embedding.grad)
        self.assertGreater(torch.count_nonzero(embedding.grad), 0)

    def test_missing_interior_state_is_rejected(self):
        states = {100: _state(0, 1), 102: _state(0, 3)}
        with self.assertRaisesRegex(ValueError, "missing an interior"):
            plan_bptt_windows(states, {102}, horizon=2)


class ProjectionCheckpointTest(unittest.TestCase):
    def test_recomputation_matches_values_and_all_parameter_gradients(self):
        for bias in (False, True):
            torch.manual_seed(10)
            hidden = torch.randn(11, 5, requires_grad=True)
            head = torch.nn.Linear(5, 257, bias=bias)
            embedding = torch.randn(257, 5, requires_grad=True)
            parameters = [hidden, head.weight, embedding] + ([head.bias] if bias else [])
            reference = torch.softmax(head(hidden).float(), dim=-1) @ embedding
            reference_grads = torch.autograd.grad(reference.square().sum(), parameters)
            for checkpointed in (False, True):
                output = hidden_to_soft_embedding(hidden, lm_head=head,
                    embedding_weight=embedding, chunk_size=3,
                    checkpoint_projection=checkpointed)
                gradients = torch.autograd.grad(output.square().sum(), parameters)
                torch.testing.assert_close(output, reference)
                for actual, expected in zip(gradients, reference_grads):
                    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)

    def test_checkpoint_does_not_retain_vocabulary_activations(self):
        torch.manual_seed(10)
        hidden = torch.randn(32, 8, requires_grad=True)
        head = torch.nn.Linear(8, 1024, bias=False)
        embedding = torch.randn(1024, 8, requires_grad=True)
        parameter_storages = {p.untyped_storage().data_ptr() for p in (head.weight, embedding)}
        saved_bytes = []
        for checkpointed in (False, True):
            saved = {}
            def pack(tensor):
                ptr = tensor.untyped_storage().data_ptr()
                if ptr not in parameter_storages:
                    saved[ptr] = tensor.untyped_storage().nbytes()
                return tensor
            with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
                result = hidden_to_soft_embedding(hidden, lm_head=head,
                    embedding_weight=embedding, chunk_size=8,
                    checkpoint_projection=checkpointed)
            saved_bytes.append(sum(saved.values()))
            result.sum().backward()
        self.assertLess(saved_bytes[1], saved_bytes[0] / 10)
        print(f"Projection saved activation bytes: {saved_bytes[0]} -> {saved_bytes[1]}")

    def test_empty_and_inference_projection(self):
        head = torch.nn.Linear(5, 7)
        embedding = torch.randn(7, 5)
        with torch.no_grad():
            hidden = torch.randn(3, 5)
            result = hidden_to_soft_embedding(hidden, lm_head=head, embedding_weight=embedding)
            torch.testing.assert_close(result, head(hidden).softmax(-1) @ embedding)
            empty = hidden_to_soft_embedding(hidden[:0], lm_head=head, embedding_weight=embedding)
            self.assertEqual(empty.shape, (0, 5))


class CompactBPTTTest(unittest.TestCase):
    @staticmethod
    def actor_method():
        # Execute the actual orchestration method without CUDA/Ray imports.
        path = _MODULE_PATH.parents[2] / "workers" / "actor" / "dp_actor.py"
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DataParallelPPOActor")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_forward_micro_batch_with_bptt")
        namespace = {"torch": torch, "expand_bptt_block_signal": _MODULE.expand_bptt_block_signal,
                     "StagedBPTTReplay": _MODULE.StagedBPTTReplay,
                     "bptt_token_validity": _MODULE.bptt_token_validity}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
        return namespace[method.name]

    def test_blocks_restore_partial_tail_and_gradients(self):
        values = torch.randn(3, 4, 5, requires_grad=True)
        starts = torch.tensor([0, 6, 9])
        lengths = torch.tensor([4, 3, 1])
        restored = _MODULE.expand_bptt_block_signal(values, starts, lengths, response_length=10)
        reference = torch.zeros(3, 10, 5)
        for row, (start, length) in enumerate(zip(starts, lengths)):
            reference[row, start:start + length] = values[row, :length]
        torch.testing.assert_close(restored, reference)
        restored.sum().backward()
        expected = torch.zeros_like(values)
        for row, length in enumerate(lengths):
            expected[row, :length] = 1
        torch.testing.assert_close(values.grad, expected)

    def test_compact_actor_matches_dense_values_and_temporal_gradients(self):
        for depth in (1, 2):
            with self.subTest(depth=depth):
                torch.manual_seed(3)
                batch, length, block, hidden = 3, 9, 3, 4
                starts = torch.tensor([0, 3, 6])
                lengths = torch.tensor([3, 3, 2])
                endpoint = torch.randn(batch, block, hidden)
                predecessors = torch.randn(batch, depth, block, hidden)
                valid = torch.ones(batch, depth, dtype=torch.bool)
                valid[0] = False
                valid[1, :-1] = False
                base = {
                    "input_ids": torch.zeros(batch, length, dtype=torch.long),
                    "responses": torch.ones(batch, length, dtype=torch.long),
                    "bptt_input_ids": torch.zeros(batch, depth, length, dtype=torch.long),
                    "bptt_responses": torch.ones(batch, depth, length, dtype=torch.long),
                    "bptt_valid_mask": valid,
                }
                compact = dict(base, bptt_rollout_self_conditioning_signal=predecessors,
                    bptt_block_starts=starts[:, None].expand(-1, depth),
                    bptt_block_lengths=lengths[:, None].expand(-1, depth),
                    rollout_self_conditioning_signal_block_values=endpoint,
                    rollout_self_conditioning_signal_block_starts=starts,
                    rollout_self_conditioning_signal_block_lengths=lengths)
                dense = dict(base,
                    rollout_self_conditioning_signal=_MODULE.expand_bptt_block_signal(endpoint, starts, lengths, response_length=length),
                    bptt_rollout_self_conditioning_signal=torch.stack([
                        _MODULE.expand_bptt_block_signal(predecessors[:, i], starts, lengths, response_length=length)
                        for i in range(depth)], dim=1))
                results = []
                for inputs in (dense, compact):
                    weight = torch.tensor(0.7, requires_grad=True)
                    bridge_outputs = []
                    def forward(data, **kwargs):
                        signal = data.get("_bptt_self_conditioning_signal", data["rollout_self_conditioning_signal"])
                        if data.get("_bptt_return_signal_only"):
                            output = signal * weight
                            output.retain_grad()
                            bridge_outputs.append(output)
                            return {"bptt_self_conditioning_signal": output}
                        return {"loss": (signal * weight).square().sum(), "generation_jsd_metrics": {}}
                    actor = types.SimpleNamespace(_forward_micro_batch=forward, config={})
                    result = self.actor_method()(actor, inputs, temperature=1.0)
                    result["loss"].backward()
                    self.assertGreater(bridge_outputs[-1].grad[2].abs().sum().item(), 0)
                    self.assertEqual(bridge_outputs[-1].grad[0].abs().sum().item(), 0)
                    results.append((result["loss"], weight.grad))
                for dense_value, compact_value in zip(*results):
                    torch.testing.assert_close(dense_value, compact_value)

    def test_missing_payload_cannot_silently_disable_bptt(self):
        actor = types.SimpleNamespace(config={"generation_consistency_jsd": {"bptt": {"enable": True}}})
        with self.assertRaisesRegex(RuntimeError, "lost its predecessor payload"):
            self.actor_method()(actor, {}, temperature=1.0)

    def test_trainer_block_builder_matches_original_dense_builder(self):
        path = _MODULE_PATH.with_name("ray_trainer.py")
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RayPPOTrainer")
        names = {"_build_round_self_conditioning_signal", "_build_block_local_replay_field"}
        methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
        namespace = {"torch": torch, "Optional": __import__("typing").Optional, "Any": object}
        for method in methods:
            method.decorator_list = []
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace)
        for start, length in ((0, 4), (4, 4), (8, 2)):
            item = {"position": (start, start + length),
                    "self_conditioning_signal": torch.randn(length, 5),
                    "valid": torch.arange(length) % 2 == 0}
            actor = types.SimpleNamespace(
                _get_self_conditioning_item=lambda states, key: states[key],
                _replay_dst_start=lambda position, **kwargs: position[0])
            valid = torch.zeros(3, 4, dtype=torch.bool)
            valid[start // 4, :length] = True
            dense = namespace["_build_round_self_conditioning_signal"](
                actor, torch.zeros(3, 4), torch.full((3, 1), 7), valid,
                {7: item}, 0, 10, 4, 5, torch.float32, torch.device("cpu"))
            block = namespace["_build_block_local_replay_field"](
                item, field_name="self_conditioning_signal", src_start=0,
                copy_len=length, block_size=4, hidden_size=5,
                dtype=torch.float32, device=torch.device("cpu"))
            reconstructed = _MODULE.expand_bptt_block_signal(
                block[None], torch.tensor([start]), torch.tensor([length]), response_length=10)
            torch.testing.assert_close(dense, reconstructed[0])

    def test_real_geometry_storage_reduction(self):
        # Failed step had 3713 replay rows; LLaDA2 uses 2048 hidden dimensions.
        dense_bytes = 3713 * 4096 * 2048 * 2
        compact_bytes = 3713 * 32 * 2048 * 2
        self.assertEqual(dense_bytes // compact_bytes, 128)
        self.assertLess(3 * compact_bytes, 1.5 * 1024**3)


class StagedBPTTTest(unittest.TestCase):
    def test_staged_matches_connected_gradients_rng_and_accumulation(self):
        for depth in (1, 2, 3):
            for stochastic in (False, True):
                with self.subTest(depth=depth, stochastic=stochastic):
                    torch.manual_seed(19)
                    batch, length, hidden = 3, 5, 4
                    valid = torch.ones(batch, depth, dtype=torch.bool)
                    valid[0] = False
                    valid[1, :-1] = False
                    inputs = {
                        "input_ids": torch.zeros(batch, length, dtype=torch.long),
                        "responses": torch.ones(batch, length, dtype=torch.long),
                        "bptt_input_ids": torch.zeros(batch, depth, length, dtype=torch.long),
                        "bptt_responses": torch.ones(batch, depth, length, dtype=torch.long),
                        "bptt_valid_mask": valid,
                        "rollout_self_conditioning_signal": torch.randn(batch, length, hidden),
                        "bptt_rollout_self_conditioning_signal": torch.randn(batch, depth, length, hidden),
                    }
                    results = []
                    for staged in (False, True):
                        torch.manual_seed(23)
                        model = torch.nn.Linear(hidden, hidden)
                        head = torch.nn.Linear(hidden, 17, bias=False)
                        embedding = torch.nn.Parameter(torch.randn(17, hidden))
                        params = list(model.parameters()) + list(head.parameters()) + [embedding]
                        optimizer = torch.optim.Adam(params, lr=0.01)
                        calls = []
                        def forward(data, **kwargs):
                            calls.append(torch.is_grad_enabled())
                            signal = data.get("_bptt_self_conditioning_signal", data["rollout_self_conditioning_signal"])
                            hidden_states = model(signal).tanh()
                            if stochastic:
                                hidden_states = torch.nn.functional.dropout(hidden_states, p=0.25)
                            if data.get("_bptt_return_signal_only"):
                                value = hidden_to_soft_embedding(hidden_states.flatten(0, 1),
                                    lm_head=head, embedding_weight=embedding, chunk_size=3)
                                return {"bptt_self_conditioning_signal": value.view(batch, length, hidden)}
                            return {"loss": head(hidden_states).square().mean(), "generation_jsd_metrics": {}}
                        actor = types.SimpleNamespace(_forward_micro_batch=forward, config={})
                        snapshots = []
                        for update in range(2):
                            optimizer.zero_grad(set_to_none=True)
                            for micro in range(2):
                                result = CompactBPTTTest.actor_method()(actor, inputs, temperature=1,
                                    staged_backward=staged)
                                # Includes normalization and a simulated AMP scale;
                                # the temporal VJP must not apply either twice.
                                (result["loss"] * 0.125 * 32).backward()
                                replay = result.pop("_bptt_backward_replay", None)
                                if replay is not None:
                                    replay.backward()
                                    self.assertEqual(replay.records, [])
                                    self.assertIsNone(replay.bridge)
                                snapshots.append((result["loss"].detach(), [p.grad.clone() for p in params]))
                            optimizer.step()
                        if staged:
                            self.assertEqual(calls, ([False] * depth + [True] * (depth + 1)) * 4)
                        results.append((snapshots, [p.detach().clone() for p in params], torch.get_rng_state()))
                    for (loss_a, grads_a), (loss_b, grads_b) in zip(results[0][0], results[1][0]):
                        torch.testing.assert_close(loss_a, loss_b)
                        for a, b in zip(grads_a, grads_b):
                            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
                    for a, b in zip(results[0][1], results[1][1]):
                        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
                    torch.testing.assert_close(results[0][2], results[1][2], rtol=0, atol=0)

    def test_staged_inference_does_not_create_backward_tape(self):
        inputs = {
            "input_ids": torch.zeros(1, 2, dtype=torch.long),
            "responses": torch.ones(1, 2, dtype=torch.long),
            "bptt_input_ids": torch.zeros(1, 1, 2, dtype=torch.long),
            "bptt_responses": torch.ones(1, 1, 2, dtype=torch.long),
            "bptt_valid_mask": torch.ones(1, 1, dtype=torch.bool),
            "rollout_self_conditioning_signal": torch.ones(1, 2, 3),
            "bptt_rollout_self_conditioning_signal": torch.ones(1, 1, 2, 3),
        }
        def forward(data, **kwargs):
            signal = data.get("_bptt_self_conditioning_signal", data["rollout_self_conditioning_signal"])
            key = "bptt_self_conditioning_signal" if data.get("_bptt_return_signal_only") else "loss"
            return {key: signal * 2}
        actor = types.SimpleNamespace(_forward_micro_batch=forward, config={})
        with torch.no_grad():
            result = CompactBPTTTest.actor_method()(actor, inputs, temperature=1, staged_backward=True)
        self.assertNotIn("_bptt_backward_replay", result)
        self.assertFalse(result["loss"].requires_grad)


if __name__ == "__main__":
    unittest.main()
