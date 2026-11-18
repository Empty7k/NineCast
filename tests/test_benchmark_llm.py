"""Valida a sequência do benchmark sem servidor ou modelos baixados."""

import argparse
import unittest
from unittest import mock

from tests import benchmark_llm


class BenchmarkLLMTests(unittest.TestCase):
    def test_memory_checkpoints_on_both_generation_paths(self):
        generation = {
            "model": "qwen2.5:3b", "done": True, "response": "Roteiro PT-BR.",
            "load_duration": 500, "prompt_eval_count": 20,
            "eval_count": 24, "eval_duration": 2_000_000_000,
        }
        for cycle in (False, True):
            with self.subTest(cycle=cycle):
                args = argparse.Namespace(model="qwen2.5:3b", num_ctx=4096, threads=6,
                                          mode="didatico", with_xtts_cycle=cycle, load_xtts=False)
                events = mock.Mock()
                events.memory.side_effect = [6400, 4000, 6300]
                events.request.side_effect = [generation] if cycle else [generation, {}]
                with mock.patch.multiple(benchmark_llm, _request=events.request,
                                         _wait_for_unload=events.wait, _xtts_cycle=events.unload_xtts,
                                         _available_mib=events.memory):
                    report = benchmark_llm._run_scenario("curto", "Material de estudo", args)
                methods = [call[0] for call in events.mock_calls]
                self.assertEqual(methods, (
                    ["unload_xtts", "memory", "request", "memory", "wait", "memory"] if cycle
                    else ["memory", "request", "memory", "request", "wait", "memory"]
                ))
                endpoint, payload = events.request.call_args_list[0].args
                self.assertEqual(endpoint, "/api/generate")
                self.assertFalse(payload["stream"])
                self.assertEqual(payload["keep_alive"], 0 if cycle else "30s")
                self.assertEqual(payload["options"]["num_gpu"], 0)
                if not cycle:
                    self.assertEqual(events.request.call_args.args, (
                        "/api/generate", {"model": args.model, "stream": False, "keep_alive": 0},
                    ))
                self.assertEqual(report["tokens_por_segundo"], 12)
                self.assertEqual(report["ram_disponivel_mib"]["apos_confirmar_liberacao"], 6300)

    def test_unload_check_retries_while_target_is_listed(self):
        other = {"model": "outro:latest"}
        with mock.patch.object(benchmark_llm, "_request", side_effect=[
            {"models": [{"model": "modelo:latest"}, other]}, {"models": [other]},
        ]) as request, mock.patch.object(benchmark_llm.time, "sleep"):
            benchmark_llm._wait_for_unload("modelo")
        self.assertEqual(request.call_args_list, [mock.call("/api/ps"), mock.call("/api/ps")])
