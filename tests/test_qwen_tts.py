"""Testes de perfis e do protocolo isolado sem importar o modelo Qwen."""

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

import config
from services import qwen_tts, qwen_tts_worker, voice


class QwenTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        patcher = mock.patch.object(config, "VOICES_DIR", self.root / "voices")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.wav = self.root / "amostra.wav"
        rate = 24000
        t = np.arange(rate) / rate
        sf.write(self.wav, 0.15 + 0.4 * np.sin(2 * np.pi * 200 * t) + 0.4 * np.sin(2 * np.pi * 20 * t), rate)

    def test_profile_published_with_restricted_permissions_and_original_hash(self):
        with mock.patch.object(qwen_tts, "_transcribe_prepared") as automatic:
            self.assertEqual(qwen_tts.register_voice("Minha voz", self.wav, "  Texto da gravação.  "),
                             "Texto da gravação.")
        automatic.assert_not_called()
        profile = config.VOICES_DIR / "qwen" / "Minha voz"
        self.assertEqual(qwen_tts.list_available_voices(), ["Minha voz"])
        self.assertEqual(profile.stat().st_mode & 0o777, 0o700)
        for file in profile.iterdir():
            self.assertEqual(file.stat().st_mode & 0o777, 0o600)
        self.assertEqual(json.loads((profile / "profile.json").read_text()), {
            "ref_text": "Texto da gravação.",
            "sha256": hashlib.sha256(self.wav.read_bytes()).hexdigest(),
            "extra_references": [],
            "reference_hashes": [hashlib.sha256(self.wav.read_bytes()).hexdigest()],
        })
        samples, rate = sf.read(profile / "reference.wav")
        self.assertEqual(rate, 24000)
        self.assertAlmostEqual(np.mean(samples), 0.0, places=5)
        self.assertAlmostEqual(np.max(np.abs(samples)), 10 ** (-1 / 20), places=5)
        t = np.arange(len(samples)) / rate
        low, speech = (abs(np.dot(samples, np.exp(-2j * np.pi * freq * t))) for freq in (20, 200))
        self.assertLess(low / speech, 0.1)
        self.assertEqual({path.name for path in profile.iterdir()}, {"profile.json", "reference.wav"})
        with self.assertRaises(FileExistsError):
            qwen_tts.register_voice("Minha voz", self.wav, "Outra transcrição.")

    def test_registration_rejects_invalid_inputs_and_keeps_no_partial_profile(self):
        for name in ("../fora", "/absoluto", ".", "voz\\fora", "nome\n"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                qwen_tts.register_voice(name, self.wav, "Texto.")
        with self.assertRaisesRegex(ValueError, "transcrição"):
            qwen_tts.register_voice("voz", self.wav, 123)
        invalid = self.root / "invalido.wav"
        invalid.write_bytes(b"arquivo invalido")
        with self.assertRaises(ValueError):
            qwen_tts.register_voice("voz", invalid, "Texto.")
        self.assertEqual(qwen_tts.list_available_voices(), [])
        self.assertEqual(list((config.VOICES_DIR / "qwen").iterdir()), [])

    def test_empty_transcript_is_generated_from_the_prepared_reference(self):
        def transcribe(path):
            self.assertEqual(path.name, "reference.wav")
            self.assertNotEqual(path.read_bytes(), self.wav.read_bytes())
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            return "Áudio transcrito."

        with mock.patch.object(qwen_tts, "_transcribe_prepared", side_effect=transcribe) as automatic:
            for index, text in enumerate((None, "", "   ")):
                name = f"Voz {index}"
                with self.subTest(text=text):
                    self.assertEqual(qwen_tts.register_voice(name, self.wav, text), "Áudio transcrito.")
                    profile = config.VOICES_DIR / "qwen" / name / "profile.json"
                    self.assertEqual(json.loads(profile.read_text())["ref_text"], "Áudio transcrito.")
        self.assertEqual(automatic.call_count, 3)

    def test_failed_automatic_transcription_does_not_publish_profile(self):
        with mock.patch.object(qwen_tts, "_transcribe_prepared", side_effect=ValueError("sem fala")):
            with self.assertRaisesRegex(ValueError, "sem fala"):
                qwen_tts.register_voice("voz", self.wav)
        self.assertEqual(qwen_tts.list_available_voices(), [])
        self.assertEqual(list((config.VOICES_DIR / "qwen").iterdir()), [])

    def test_transcribe_reference_uses_isolated_worker_and_removes_prepared_copy(self):
        process = mock.MagicMock()
        process.stdout = ['{"text": "  Frase reconhecida.  "}']
        process.wait.return_value = process.poll.return_value = 0
        prepared_paths = []

        def capture(payload):
            job = json.loads(payload)
            prepared = Path(job.pop("reference"))
            prepared_paths.append(prepared)
            self.assertNotEqual(prepared.read_bytes(), self.wav.read_bytes())
            self.assertEqual(job, {"action": "transcribe", "model": "small", "language": "pt", "threads": 6,
                                   "download_root": str(self.root / "models/whisper")})

        process.stdin.__enter__.return_value = process.stdin
        process.stdin.write.side_effect = capture
        with mock.patch.object(config, "MODELS_DIR", self.root / "models"), \
             mock.patch.object(config, "DEFAULT_LANGUAGE", "pt"), \
             mock.patch.object(config, "CPU_THREADS", 6), \
             mock.patch.object(config, "WHISPER_MODEL", "small"), \
             mock.patch.object(qwen_tts.subprocess, "Popen", return_value=process) as start:
            self.assertEqual(qwen_tts.transcribe_reference(self.wav), "Frase reconhecida.")
        self.assertEqual(start.call_args.args[0][1:3], ["-I", "-u"])
        process.wait.assert_called_once()
        process.terminate.assert_not_called()
        self.assertEqual(len(prepared_paths), 1)
        self.assertFalse(prepared_paths[0].exists())

    def test_transcription_errors_are_clear_and_worker_is_reaped(self):
        for event, kind, message in (
            ({"error": "no_speech"}, ValueError, "fala transcritível"),
            ({"error": "installation"}, RuntimeError, "faster-whisper.*--install-qwen-tts"),
        ):
            with self.subTest(event=event):
                process = mock.MagicMock()
                process.stdout = [json.dumps(event)]
                process.poll.return_value = None
                process.wait.side_effect = [subprocess.TimeoutExpired("worker", 5), -9]
                with mock.patch.object(qwen_tts.subprocess, "Popen", return_value=process):
                    with self.assertRaisesRegex(kind, message):
                        qwen_tts.transcribe_reference(self.wav)
                process.terminate.assert_called_once()
                process.kill.assert_called_once()
                self.assertEqual(process.wait.call_args_list, [mock.call(timeout=5), mock.call()])

    def test_whisper_worker_uses_int8_vad_and_consumes_segments(self):
        package = mock.Mock()
        model = package.WhisperModel.return_value
        job = {"model": "small", "reference": str(self.wav), "language": "pt",
               "threads": 6, "download_root": str(self.root / "whisper")}
        emit = mock.Mock()
        model.transcribe.return_value = (iter([mock.Mock(text=" Frase. "), mock.Mock(text=" Outra.")]), None)
        with mock.patch.dict(sys.modules, {"faster_whisper": package}):
            self.assertTrue(qwen_tts_worker.transcribe(job, emit))
            model.transcribe.return_value = (iter([]), None)
            self.assertFalse(qwen_tts_worker.transcribe(job, emit))
        package.WhisperModel.assert_called_with("small", device="cpu", compute_type="int8", cpu_threads=6,
                                                download_root=job["download_root"])
        self.assertEqual(model.transcribe.call_args.kwargs, {"language": "pt", "vad_filter": True})
        samples = model.transcribe.call_args.args[0]
        self.assertEqual((samples.shape, samples.dtype), ((16000,), np.dtype("float32")))
        self.assertEqual(emit.call_args_list, [mock.call({"text": "Frase. Outra."}), mock.call({"error": "no_speech"})])

    def test_linked_profiles_are_excluded(self):
        qwen_tts.register_voice("voz", self.wav, "Texto.")
        root = config.VOICES_DIR / "qwen"
        (root / "alias").symlink_to(root / "voz", target_is_directory=True)
        self.assertEqual(qwen_tts.list_available_voices(), ["voz"])
        with mock.patch.object(qwen_tts, "is_available", return_value=True):
            with self.assertRaises(FileNotFoundError):
                qwen_tts.synthesize_chunks(["Trecho."], "alias", [self.root / "out.wav"])

    def test_availability_detects_installed_isolated_package(self):
        venv = self.root / "isolado"
        site = venv / "lib/python3.12/site-packages/qwen_tts"
        site.mkdir(parents=True)
        (venv / "bin").mkdir()
        (venv / "bin/python").symlink_to(sys.executable)
        with mock.patch.object(config, "QWEN_TTS_VENV_DIR", venv):
            self.assertFalse(qwen_tts.is_available())
            (site / "__init__.py").touch()
            self.assertTrue(qwen_tts.is_available())

    def test_one_worker_reports_progress_while_memory_is_reserved(self):
        qwen_tts.register_voice("voz", self.wav, "Texto.")
        metadata = config.VOICES_DIR / "qwen" / "voz" / "profile.json"
        original = json.loads(metadata.read_text())
        metadata.write_text(json.dumps({key: original[key] for key in ("ref_text", "sha256")}))
        self.assertEqual(qwen_tts.list_available_voices(), ["voz"])
        paths = [self.root / "01.wav", self.root / "02.wav"]
        steps = []
        process_code = """
import json, sys, wave
job = json.load(sys.stdin)
assert job["chunks"] == ["Primeiro.", "Segundo."]
assert job["ref_text"] == "Texto."
assert job["extra_references"] == []
for i, path in enumerate(job["outputs"], 1):
    with wave.open(path, "wb") as wav:
        wav.setparams((1, 2, 24000, 0, "NONE", ""))
        wav.writeframes(b"\\0" * 4800)
    print(json.dumps({"current": i, "synthesis_seconds": 0.5}), flush=True)
"""
        popen = subprocess.Popen
        with mock.patch.object(qwen_tts, "is_available", return_value=True), \
             mock.patch.object(voice, "unload_model") as unload, \
             mock.patch.object(qwen_tts.subprocess, "Popen") as start:
            def spawn(command, **kwargs):
                unload.assert_called_once()
                self.assertEqual(command[1:3], ["-I", "-u"])
                return popen([sys.executable, "-c", process_code], **kwargs)

            start.side_effect = spawn
            result = qwen_tts.synthesize_chunks(
                ["Primeiro.", "Segundo."], "voz", paths, lambda i, n: steps.append((i, n)),
            )
            start.assert_called_once()
        self.assertEqual(steps, [(1, 2), (2, 2)])
        self.assertEqual([r["audio_duration"] for r in result], [0.1, 0.1])
        self.assertEqual([r["synthesis_seconds"] for r in result], [0.5, 0.5])

    def test_worker_is_reaped_after_callback_failure(self):
        qwen_tts.register_voice("voz", self.wav, "Texto.")
        popen = subprocess.Popen
        processes = []
        code = """
import json, sys, time
sys.stdin.read()
print(json.dumps({"current": 1, "synthesis_seconds": 0.5}), flush=True)
time.sleep(30)
"""

        def spawn(command, **kwargs):
            process = popen([sys.executable, "-c", code], **kwargs)
            processes.append(process)
            return process

        with mock.patch.object(qwen_tts, "is_available", return_value=True), \
             mock.patch.object(qwen_tts.subprocess, "Popen", side_effect=spawn):
            callback = mock.Mock(side_effect=ValueError("cancelado"))
            with self.assertRaisesRegex(ValueError, "cancelado"):
                qwen_tts.synthesize_chunks(["Trecho."], "voz", [self.root / "out.wav"], callback)
        self.assertIsNotNone(processes[0].returncode)

    def test_registers_multiple_references_and_validates_profile(self):
        sources = [self.wav, self.root / "second.wav", self.root / "third.wav"]
        for index, source in enumerate(sources[1:], 1):
            sf.write(source, np.sin(np.arange(24000) * (0.05 + index / 100)), 24000)
        for count in (1, 2, 3):
            with self.subTest(count=count):
                name = f"voz-{count}"
                self.assertEqual(qwen_tts.register_voice(name, sources[:count], "Texto."), "Texto.")
                profile = config.VOICES_DIR / "qwen" / name
                extras = [f"extra_{index:02d}.wav" for index in range(1, count)]
                metadata = json.loads((profile / "profile.json").read_text())
                self.assertEqual(metadata["extra_references"], extras)
                self.assertEqual(metadata["reference_hashes"],
                                 [hashlib.sha256(p.read_bytes()).hexdigest() for p in sources[:count]])
                for filename in ["reference.wav", *extras]:
                    self.assertEqual((profile / filename).stat().st_mode & 0o777, 0o600)
        for invalid in ([], sources + [self.wav]):
            with self.assertRaisesRegex(ValueError, "1 a 3"):
                qwen_tts.register_voice("inválida", invalid, "Texto.")

        profile = config.VOICES_DIR / "qwen" / "voz-3"
        output = self.root / "out.wav"
        with mock.patch.object(qwen_tts, "is_available", return_value=True), \
             mock.patch.object(qwen_tts, "_run_worker") as worker:
            # Substitui a inferência e verifica os caminhos numerados entregues ao worker.
            def synthesize(job, emit):
                self.assertEqual(job["extra_references"],
                                 [str(profile / "extra_01.wav"), str(profile / "extra_02.wav")])
                sf.write(output, np.zeros(2400), 24000, subtype="PCM_16")
                emit({"current": 1, "synthesis_seconds": 0.5})

            worker.side_effect = synthesize
            qwen_tts.synthesize_chunks(["Trecho."], "voz-3", [output])
            worker.reset_mock()
            extra = profile / "extra_01.wav"
            extra.unlink()
            self.assertNotIn("voz-3", qwen_tts.list_available_voices())
            extra.symlink_to(self.wav)
            self.assertNotIn("voz-3", qwen_tts.list_available_voices())
            with self.assertRaises(FileNotFoundError):
                qwen_tts.synthesize_chunks(["Trecho."], "voz-3", [output])
            worker.assert_not_called()

    def test_worker_combines_resampled_embeddings_and_preserves_icl(self):
        import torch

        extra = self.root / "extra.wav"
        sf.write(extra, np.zeros(24000), 12000)
        item = mock.Mock(ref_code=object(), ref_text="Texto principal.")
        code = item.ref_code
        package = mock.Mock()
        model = package.Qwen3TTSModel.from_pretrained.return_value
        model.create_voice_clone_prompt.return_value = [item]
        model.model.speaker_encoder_sample_rate = 24000
        model.model.extract_speaker_embedding.side_effect = [torch.tensor([3., 0.]), torch.tensor([0., 6.])]
        model.generate_voice_clone.return_value = ([np.zeros(2400)], 24000)
        job = {"threads": 6, "model": "modelo-local", "reference": str(self.wav),
               "ref_text": "Texto principal.", "extra_references": [str(extra)],
               "chunks": ["Trecho."], "outputs": [str(self.root / "out.wav")]}
        with mock.patch.dict(sys.modules, {"qwen_tts": package}), \
             mock.patch.object(torch, "set_num_threads"), mock.patch.object(torch, "set_num_interop_threads"):
            qwen_tts_worker.synthesize(job, mock.Mock())
        calls = model.model.extract_speaker_embedding.call_args_list
        self.assertEqual([(len(c.kwargs["audio"]), c.kwargs["sr"]) for c in calls],
                         [(24000, 24000), (48000, 24000)])
        expected = torch.tensor([1., 4.]) * (5 / np.sqrt(17))
        torch.testing.assert_close(item.ref_spk_embedding, expected)
        self.assertIs(item.ref_code, code)
        self.assertEqual(item.ref_text, "Texto principal.")
        self.assertIs(model.generate_voice_clone.call_args.kwargs["voice_clone_prompt"][0], item)

    def test_worker_loads_model_and_icl_prompt_once(self):
        outputs = [self.root / "01.wav", self.root / "02.wav"]
        qwen_package = mock.Mock()
        torch = mock.MagicMock()
        model = qwen_package.Qwen3TTSModel.from_pretrained.return_value
        model.generate_voice_clone.return_value = ([np.full(2400, 0.1)], 24000)
        job = {"threads": 6, "model": "modelo-local", "reference": str(self.wav),
               "ref_text": "Uma referência.", "chunks": ["Primeiro.", "Segundo."],
               "outputs": [str(path) for path in outputs]}
        emit = mock.Mock()
        with mock.patch.dict(sys.modules, {"torch": torch, "qwen_tts": qwen_package}), \
             mock.patch.dict(qwen_tts_worker.os.environ):
            qwen_tts_worker.synthesize(job, emit)
        torch.set_num_threads.assert_called_once_with(6)
        qwen_package.Qwen3TTSModel.from_pretrained.assert_called_once_with(
            "modelo-local", device_map="cpu", dtype=torch.float32, attn_implementation="sdpa",
            local_files_only=False,
        )
        model.create_voice_clone_prompt.assert_called_once()
        prompt_args = model.create_voice_clone_prompt.call_args.kwargs
        self.assertEqual(prompt_args["ref_text"], job["ref_text"])
        self.assertFalse(prompt_args["x_vector_only_mode"])
        self.assertEqual([call.args[0]["current"] for call in emit.call_args_list], [1, 2])
        for call, text in zip(model.generate_voice_clone.call_args_list, job["chunks"], strict=True):
            self.assertEqual(call.kwargs["text"], text)
            self.assertEqual(call.kwargs["language"], "Portuguese")
            self.assertEqual(call.kwargs["temperature"], 0.9)
            self.assertIs(call.kwargs["voice_clone_prompt"], model.create_voice_clone_prompt.return_value)
        for path in outputs:
            info = sf.info(path)
            self.assertEqual((info.samplerate, info.subtype, info.channels), (24000, "PCM_16", 1))
