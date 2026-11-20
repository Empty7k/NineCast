"""Orquestração local do Ollama sem servidor, pesos ou síntese real."""

import io
import unittest
from unittest.mock import Mock, call, patch
from contextlib import nullcontext
from urllib.error import HTTPError, URLError

import config
from services import llm, voice


def response(script, reason="stop"):
    return {"response": script, "done": True, "done_reason": reason}


class StudyScriptTests(unittest.TestCase):
    def setUp(self):
        self.events = Mock()
        self.client = self.events.client
        self.client.serving.return_value = nullcontext()
        self.client.require_local_model.return_value = {}
        self.client_type = self.enterContext(patch.object(llm, "LocalOllama", return_value=self.client))
        self.enterContext(patch.object(voice, "unload_model", self.events.unload_xtts))
        self.enterContext(patch.object(config, "OLLAMA_MODEL", "modelo:teste"))
        self.enterContext(patch.object(config, "LLM_MAX_SECTION_CHARS", 20))

    def test_invalid_inputs_are_rejected_before_loading_models(self):
        with patch.object(config, "MAX_TEXT_CHARS", 12):
            for args in ({"text": " \n "}, {"text": " " * 12 + "x"}, {"text": "x", "mode": "outro"},
                         {"text": "x", "model": " "}, {"text": "x", "progress_callback": 123}):
                with self.subTest(args=args), self.assertRaises(ValueError):
                    llm.generate_study_script(**args)
        self.events.unload_xtts.assert_not_called()
        self.client_type.assert_not_called()

    def test_single_section_releases_xtts_before_api_and_uses_inline_unload(self):
        self.client.request.return_value = response("  Um roteiro.  ")
        result = llm.generate_study_script("Material.", "resumo", "outro", self.events.progress)
        self.assertEqual([event[0] for event in self.events.mock_calls], [
            "unload_xtts", "client.serving", "client.require_local_model", "client.request",
            "progress", "client.wait_for_unload",
        ])
        self.assertEqual(result["script"], "Um roteiro.")
        self.assertEqual((result["model"], result["mode"], result["section_count"]), ("outro", "resumo", 1))
        self.assertFalse(result["truncated"])
        self.assertGreaterEqual(result["elapsed_seconds"], 0)
        self.events.progress.assert_called_once_with(1, 1)
        payload = self.client.request.call_args.args[1]
        self.assertEqual(payload["prompt"], "Material.")
        self.assertEqual(payload["keep_alive"], 0)
        self.assertFalse(payload["stream"])
        self.assertIn(llm.MODES["resumo"], payload["system"])
        self.assertIn("Não realize cálculos", payload["system"])

    def test_disables_thinking_and_strips_reasoning_from_the_script(self):
        self.client.require_local_model.return_value = {"capabilities": ["thinking"]}
        self.client.request.return_value = response("Começo. <THINK>oculto</think> Final.<think>cauda")
        with patch.object(config, "LLM_TIMEOUT_SECONDS", 700):
            result = llm.generate_study_script("Material.", model="qwen3:4b")
        self.assertIs(self.client.request.call_args.args[1]["think"], False)
        self.assertEqual(self.client.request.call_args.kwargs["timeout"], 700)
        self.assertEqual(result["script"], "Começo.  Final.")

    def test_rejects_thinking_only_variant_before_sending_material(self):
        self.client.require_local_model.return_value = {"thinking": {"values": [True]}}
        with self.assertRaisesRegex(ValueError, "exige raciocínio"):
            llm.generate_study_script("Texto de estudo.", model="qwen3:4b")
        self.client.request.assert_not_called()

    def test_caps_context_for_advanced_model_with_native_thinking_control(self):
        self.client.require_local_model.return_value = {"thinking": {"values": [False, True]}}
        self.client.request.return_value = response("Um roteiro.")
        with patch.object(llm, "_available_memory_bytes", return_value=12 * 2**30):
            llm.generate_study_script("Texto de estudo.", model="qwen3:14b")
        payload = self.client.request.call_args.args[1]
        self.assertIs(payload["think"], False)
        self.assertEqual(payload["options"]["num_ctx"], 2048)

    def test_advanced_model_checks_available_ram_before_generation(self):
        with patch.object(llm, "_available_memory_bytes", return_value=8 * 2**30), \
             self.assertRaisesRegex(RuntimeError, "RAM disponível"):
            llm.generate_study_script("Texto de estudo.", model="qwen3:14b")
        self.client.request.assert_not_called()

    def test_sections_preserve_boundaries_and_report_any_length_cutoff(self):
        source = "Um parágrafo.\n\nFrase dois. Frase três."
        self.client.request.side_effect = [response("Um.", "length"), response("Dois."), response("Três."), {}]
        result = llm.generate_study_script(source, progress_callback=self.events.progress)
        payloads = [item.args[1] for item in self.client.request.call_args_list]
        self.assertEqual([p["prompt"] for p in payloads[:-1]], ["Um parágrafo.", "Frase dois.", "Frase três."])
        self.assertTrue(all(p["keep_alive"] == "30s" for p in payloads[:-1]))
        self.assertEqual(payloads[-1], {"model": "modelo:teste", "stream": False, "keep_alive": 0})
        self.assertEqual(result["script"], "Um.\n\nDois.\n\nTrês.")
        self.assertEqual(result["section_count"], 3)
        self.assertTrue(result["truncated"])
        self.assertEqual(self.events.progress.call_args_list, [call(1, 3), call(2, 3), call(3, 3)])
        self.client.wait_for_unload.assert_called_once_with("modelo:teste")

    def test_second_section_failure_still_unloads(self):
        self.client.request.side_effect = [response("Um."), RuntimeError("Falha na geração"), {}]
        with self.assertRaisesRegex(RuntimeError, "Falha na geração"):
            llm.generate_study_script("Primeiro parágrafo.\nSegundo parágrafo.", progress_callback=self.events.progress)
        self.events.progress.assert_called_once_with(1, 2)
        self.assertEqual(self.client.request.call_args.args[1]["keep_alive"], 0)
        self.client.wait_for_unload.assert_called_once_with("modelo:teste")

    def test_unload_failure_preserves_original_error(self):
        self.client.request.side_effect = [RuntimeError("geração"), RuntimeError("liberação")]
        with self.assertRaisesRegex(RuntimeError, "geração") as error:
            llm.generate_study_script("Primeiro parágrafo.\nSegundo parágrafo.")
        self.assertIn("liberação", error.exception.__notes__[0])

    def test_sections_fall_back_to_words_and_long_identifiers(self):
        self.assertEqual(llm._sections("abc def ghi", 7), ["abc def", "ghi"])
        self.assertEqual(llm._sections("0123456789", 4), ["0123", "4567", "89"])

    def test_removes_citations_before_splitting_and_keeps_code(self):
        self.client.request.return_value = response("Roteiro.")
        llm.generate_study_script("Use COUNT(*) (SILVA, 2020, p. 42).")
        self.assertEqual(self.client.request.call_args.args[1]["prompt"], "Use COUNT(*).")
        with self.assertRaisesRegex(ValueError, "referências"):
            llm.generate_study_script("(LIMA, 2022, p. 18)")


class LocalOllamaTests(unittest.TestCase):
    def test_accepts_only_http_loopback_origins(self):
        for url in ("http://localhost", "http://[::1]:11434/", "http://127.0.0.1:11434"):
            self.assertEqual(llm.LocalOllama(url).base_url, url.rstrip("/"))
        for url in ("https://localhost", "http://0.0.0.0", "http://localhost.exemplo.org",
                    "http://localhost@exemplo.org", "http://localhost:65536", "http://localhost/api"):
            with self.subTest(url=url), patch.object(config, "OLLAMA_BASE_URL", url), \
                    patch.object(voice, "unload_model") as unload, self.assertRaises(ValueError):
                llm.generate_study_script("Aula.")
            unload.assert_not_called()

    def test_remote_alias_is_rejected_before_sending_text(self):
        with patch.object(llm.LocalOllama, "request", side_effect=[{"models": []}, {"remote_host": "remoto"}]) as request, \
                patch.object(voice, "unload_model"), self.assertRaisesRegex(ValueError, "locais"):
            llm.generate_study_script("Material privado.", model="apelido")
        self.assertEqual(request.call_args_list, [call("/api/ps", timeout=2), call("/api/show", {"model": "apelido"})])

    def test_actionable_api_errors(self):
        client = llm.LocalOllama("http://localhost:11434")
        for error, message in (
            (URLError(ConnectionRefusedError()), "ollama serve"),
            (URLError(TimeoutError()), "tempo de espera"),
            (TimeoutError(), "tempo de espera"),
            (HTTPError(client.base_url, 404, "Not Found", {}, io.BytesIO()), "ollama pull meu-modelo"),
            (HTTPError(client.base_url, 307, "Redirect", {}, io.BytesIO()), "HTTP 307"),
        ):
            with self.subTest(error=error), patch.object(client._http, "open", side_effect=error), \
                    self.assertRaisesRegex(RuntimeError, message):
                client.require_local_model("meu-modelo")

    def test_reuses_running_server(self):
        client = llm.LocalOllama("http://localhost:11434")
        with patch.object(client, "request", return_value={"models": []}) as request, \
             patch.object(llm.subprocess, "Popen") as start, client.serving():
            request.assert_called_once_with("/api/ps", timeout=2)
            start.assert_not_called()

    def test_starts_owned_server_and_stops_it_even_if_generation_fails(self):
        client = llm.LocalOllama("http://localhost:11434")
        for error in (None, ValueError("Falha na geração")):
            with self.subTest(error=error), \
                 patch.object(client, "request", side_effect=[llm._OllamaOffline(), {"models": []}]), \
                 patch.object(llm.shutil, "which", return_value="/usr/bin/ollama"), \
                 patch.object(llm.subprocess, "Popen") as start:
                process = start.return_value
                process.poll.return_value = None
                try:
                    with client.serving():
                        self.assertEqual(start.call_args.args[0], ["/usr/bin/ollama", "serve"])
                        self.assertEqual(start.call_args.kwargs["env"]["OLLAMA_HOST"], "127.0.0.1:11434")
                        self.assertEqual(start.call_args.kwargs["env"]["OLLAMA_NO_CLOUD"], "1")
                        if error:
                            raise error
                except ValueError as exc:
                    self.assertIs(exc, error)
                process.terminate.assert_called_once_with()
                process.wait.assert_called_once_with(timeout=5)

    def test_offline_listing_and_cloud_exclusion(self):
        models = [{"name": "qwen3:4b", "size": 2_498_000_000},
                  {"name": "remote", "size": 100, "remote_host": "example"},
                  {"name": "alias:cloud", "size": 100},
                  {"name": "broken"}]
        with patch.object(llm.LocalOllama, "request", return_value={"models": models}):
            self.assertEqual(llm.list_local_models(), [{"name": "qwen3:4b", "size_bytes": 2_498_000_000}])
        with patch.object(llm.LocalOllama, "request", side_effect=llm._OllamaOffline):
            self.assertEqual(llm.list_local_models(), [])

    def test_kills_owned_server_only_after_termination_timeout(self):
        client = llm.LocalOllama("http://[::1]:11434")
        with patch.object(client, "request", side_effect=[llm._OllamaOffline(), {"models": []}]), \
             patch.object(llm.shutil, "which", return_value="/usr/bin/ollama"), \
             patch.object(llm.subprocess, "Popen") as start:
            process = start.return_value
            process.poll.return_value = None
            process.wait.side_effect = [llm.subprocess.TimeoutExpired("ollama", 5), 0]
            with client.serving():
                self.assertEqual(start.call_args.kwargs["env"]["OLLAMA_HOST"], "[::1]:11434")
            self.assertEqual(process.method_calls[-4:], [
                call.terminate(), call.wait(timeout=5), call.kill(), call.wait(),
            ])

    def test_startup_reports_stderr_and_reaps_exited_process(self):
        client = llm.LocalOllama("http://localhost:11434")
        process = Mock()
        process.poll.return_value = 1

        def start(*args, stderr, **kwargs):
            stderr.write(b"porta ocupada")
            return process

        with patch.object(client, "request", side_effect=llm._OllamaOffline), \
             patch.object(llm.shutil, "which", return_value="/usr/bin/ollama"), \
             patch.object(llm.subprocess, "Popen", side_effect=start), \
             self.assertRaisesRegex(RuntimeError, "porta ocupada"):
            with client.serving():
                self.fail("Servidor não iniciou")
        process.terminate.assert_not_called()
        process.wait.assert_called_once_with(timeout=5)

    def test_catalog_flags_large_models_and_unknown_sizes(self):
        self.assertIsNone(llm.describe_model("qwen3:4b")["alerta_memoria"])
        self.assertTrue(llm.describe_model("qwen3:14b")["alerta_memoria"])
        self.assertEqual(llm.describe_model("outro", 9_000_000_000)["tier"], "Avançado")
        self.assertEqual(llm.describe_model("outro")["ram_estimada"], "Desconhecida")
