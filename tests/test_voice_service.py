"""Testes rápidos; adicione --integration --agree-license para executar o XTTS real."""

import argparse
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config
from services import voice


class FakeModel:
    def __init__(self):
        self.args = SimpleNamespace(gpt_max_text_tokens=400)
        self.tokenizer = SimpleNamespace(encode=lambda text, lang: list(text))
        self.extractions = 0

    def get_speaker_embedding(self, signal, rate):
        return torch.ones(1, 3, 1)

    def get_gpt_cond_latents(self, signal, rate, **kwargs):
        self.extractions += 1
        return torch.ones(1, 2)

    def inference(self, **kwargs):
        return {"wav": np.full(24000, 0.1, dtype=np.float32)}


class ModelLifecycleTests(unittest.TestCase):
    def test_unload_clears_cache_and_loading_again_creates_new_model(self):
        first, second = FakeModel(), FakeModel()
        api = SimpleNamespace(TTS=mock.Mock(side_effect=[
            mock.Mock(**{"to.return_value": SimpleNamespace(synthesizer=SimpleNamespace(tts_model=first))}),
            mock.Mock(**{"to.return_value": SimpleNamespace(synthesizer=SimpleNamespace(tts_model=second))}),
        ]))
        with mock.patch.dict(sys.modules, {"TTS.api": api}), mock.patch.dict(os.environ), \
             mock.patch.object(voice.torch, "set_num_threads"), mock.patch.object(voice, "_model", None), \
             mock.patch.object(voice.gc, "collect") as collect, \
             mock.patch.object(voice.ctypes, "CDLL") as libc:
            self.assertFalse(voice.unload_model())
            self.assertIs(voice._get_model(), first)
            self.assertTrue(voice.unload_model())
            self.assertIsNone(voice._model)
            self.assertFalse(voice.unload_model())
            collect.assert_called_once_with()
            libc.assert_called_once_with("libc.so.6")
            libc.return_value.malloc_trim.assert_called_once_with(0)
            self.assertIs(voice._get_model(), second)
            self.assertIs(voice._get_model(), second)
            self.assertEqual(api.TTS.call_count, 2)

    def test_unload_works_without_glibc_or_malloc_trim(self):
        for error in (OSError(), AttributeError()):
            with self.subTest(error=error), mock.patch.object(voice, "_model", FakeModel()), \
                 mock.patch.object(voice.gc, "collect"), mock.patch.object(voice.ctypes, "CDLL", side_effect=error):
                self.assertTrue(voice.unload_model())
                self.assertIsNone(voice._model)


class ConditioningTests(unittest.TestCase):
    def test_duration_weighting_normalizes_channels_and_trims_only_style(self):
        model = mock.Mock()
        model.get_speaker_embedding.side_effect = [
            torch.tensor([[[1.], [0.]]]), torch.tensor([[[0.], [1.]]]),
        ]
        rate = 100
        signals = [torch.ones(1, 175), torch.ones(1, 525)]
        style, speaker = voice._pool_reference_latents(model, signals, rate)
        self.assertIs(style, model.get_gpt_cond_latents.return_value)
        self.assertEqual(model.get_gpt_cond_latents.call_args.args[0].shape, (1, 600))
        self.assertEqual(model.get_speaker_embedding.call_count, 2)
        torch.testing.assert_close(speaker.flatten(), torch.tensor([1., 3.]) / 10 ** .5)
        torch.testing.assert_close(speaker.norm(dim=1), torch.ones(1, 1))

    def test_style_keeps_short_single_reference_or_two_second_tail(self):
        for samples in (100, 600, 800):
            with self.subTest(samples=samples):
                model = mock.Mock()
                model.get_speaker_embedding.return_value = torch.ones(1, 2, 1)
                voice._pool_reference_latents(model, [torch.ones(1, samples)], 100)
                self.assertEqual(model.get_gpt_cond_latents.call_args.args[0].size(-1), samples)

    def test_filter_reduces_rumble_before_extracting_both_latents(self):
        rate = 24000
        time = np.arange(rate) / rate
        samples = 0.2 * (np.sin(2 * np.pi * 30 * time) + np.sin(2 * np.pi * 200 * time))
        model = mock.Mock()
        model.get_speaker_embedding.return_value = torch.ones(1, 2, 1)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tons.wav"
            sf.write(path, samples, rate, subtype="FLOAT")
            voice._extract_latents(model, [path])
        signal, conditioning_rate = model.get_speaker_embedding.call_args.args
        self.assertEqual(conditioning_rate, 22050)
        self.assertAlmostEqual(signal.abs().max().item(), 0.75, places=5)
        spectrum = np.abs(np.fft.rfft(signal.numpy()[0]))
        self.assertLess(spectrum[30] / spectrum[200], 0.2)
        torch.testing.assert_close(model.get_gpt_cond_latents.call_args.args[0], signal)


class VoiceServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.patch_dir = mock.patch.object(config, "VOICES_DIR", self.root / "voices")
        self.patch_dir.start()
        self.addCleanup(self.patch_dir.stop)
        config.VOICES_DIR.mkdir()
        self.reference = self.root / "reference.wav"
        t = np.arange(24000, dtype=np.float32) / 24000
        sf.write(self.reference, 0.25 * np.sin(2 * np.pi * 180 * t), 24000)
        self.fake = FakeModel()
        self.model_patch = mock.patch.object(voice, "_get_model", return_value=self.fake)
        self.model_patch.start()
        self.addCleanup(self.model_patch.stop)

    def test_create_synthesize_twice_and_reuse_cache(self):
        voice.get_or_create_voice_latents("teste", [self.reference])
        self.assertEqual(self.fake.extractions, 1)
        self.assertEqual(voice.list_available_voices(), ["teste"])
        with mock.patch.object(voice, "_prepare_reference", side_effect=AssertionError("reprocessou")), \
             mock.patch.object(self.fake, "inference", wraps=self.fake.inference) as infer:
            for index in range(2):
                result = voice.synthesize_speech("Olá, este é um teste.", "teste", self.root / f"{index}.wav")
                self.assertEqual(result["audio_duration"], 1.0)
                self.assertGreaterEqual(result["rtf"], 0)
            voice.get_or_create_voice_latents("teste", [self.reference])
        self.assertEqual(self.fake.extractions, 1)
        self.assertEqual(sf.info(self.root / "1.wav").samplerate, 24000)
        params = infer.call_args.kwargs
        self.assertEqual((params["temperature"], params["top_p"], params["top_k"], params["repetition_penalty"]),
                         (0.70, 0.85, 50, 6.0))

    def test_changed_reference_invalidates_cache(self):
        voice.get_or_create_voice_latents("teste", [self.reference])
        t = np.arange(24000) / 24000
        sf.write(self.reference, 0.1 * np.sin(2 * np.pi * 220 * t), 24000)
        voice.get_or_create_voice_latents("teste", [self.reference])
        self.assertEqual(self.fake.extractions, 2)

    def test_v2_cache_is_upgraded_only_when_references_are_available(self):
        voice.get_or_create_voice_latents("teste", [self.reference])
        cache = config.VOICES_DIR / "teste.latents.pt"
        data = torch.load(cache, weights_only=True)
        data["preparation_version"] = 2
        data["speaker_embedding"] = torch.zeros(1, 3, 1)
        torch.save(data, cache)
        _, previous = voice.get_or_create_voice_latents("teste")
        self.assertEqual(previous.norm().item(), 0)
        _, updated = voice.get_or_create_voice_latents("teste", [self.reference])
        torch.testing.assert_close(updated.norm(), torch.tensor(1.0))
        self.assertEqual(torch.load(cache, weights_only=True)["preparation_version"], 3)

    def test_legacy_cache_is_usable_and_rebuilt_when_references_are_supplied(self):
        voice.get_or_create_voice_latents("teste", [self.reference])
        cache = config.VOICES_DIR / "teste.latents.pt"
        data = torch.load(cache, weights_only=True)
        del data["preparation_version"]
        torch.save(data, cache)
        voice.get_or_create_voice_latents("teste")
        self.assertEqual(self.fake.extractions, 1)
        voice.get_or_create_voice_latents("teste", [self.reference])
        self.assertEqual(self.fake.extractions, 2)

    def test_friendly_errors_and_no_overwrite(self):
        with self.assertRaisesRegex(ValueError, "texto"):
            voice.synthesize_speech(" ", "teste", self.root / "audio.wav")
        with self.assertRaisesRegex(FileNotFoundError, "referências"):
            voice.synthesize_speech("Olá", "inexistente", self.root / "audio.wav")
        with self.assertRaises(ValueError):
            voice.get_or_create_voice_latents("../fora", [self.reference])
        voice.get_or_create_voice_latents("teste", [self.reference])
        existing = self.root / "existing.wav"
        existing.write_bytes(b"nao sobrescrever")
        with self.assertRaises(FileExistsError):
            voice.synthesize_speech("Olá", "teste", existing)
        self.assertEqual(existing.read_bytes(), b"nao sobrescrever")

    def test_memory_error_message(self):
        with mock.patch.object(voice, "_prepare_reference", side_effect=MemoryError):
            with self.assertRaisesRegex(MemoryError, "Memória insuficiente"):
                voice.get_or_create_voice_latents("teste", [self.reference])

    def test_short_silences_are_preserved_and_long_ones_shortened(self):
        # Intervalos determinísticos evitam depender da resolução de quadros do librosa.
        rate = 24000
        t = np.arange(rate * 3) / rate
        sf.write(self.reference, 0.1 * np.sin(2 * np.pi * 180 * t), rate)
        intervals = np.array([[0, 12000], [18000, 30000], [42000, 54000]])
        output = self.root / "prepared.wav"
        with mock.patch("librosa.effects.split", return_value=intervals):
            voice._prepare_reference(self.reference, output)
        prepared, _ = sf.read(output)
        # A pausa de 0,25 s é preservada; a de 0,5 s vira 0,15 s mais padding.
        self.assertAlmostEqual(len(prepared) / rate, 1.99, places=2)

    def test_reference_dc_removal_and_safe_peak_normalization(self):
        rate = 24000
        t = np.arange(rate) / rate
        for amplitude, offset in ((0.02, 0.15), (0.7, -0.2)):
            with self.subTest(amplitude=amplitude):
                output = self.root / "prepared.wav"
                sf.write(self.reference, offset + amplitude * np.sin(2 * np.pi * 180 * t),
                         rate, subtype="FLOAT")
                original = self.reference.read_bytes()
                voice._prepare_reference(self.reference, output)
                prepared, prepared_rate = sf.read(output)
                self.assertEqual(prepared_rate, rate)
                self.assertTrue(np.isfinite(prepared).all())
                self.assertLess(abs(np.mean(prepared)), 1e-6)
                self.assertAlmostEqual(np.max(np.abs(prepared)), 10 ** (-1 / 20), places=6)
                self.assertEqual(self.reference.read_bytes(), original)

    def test_reference_rejects_dc_only_and_silence(self):
        for level in (0.0, 0.25):
            with self.subTest(level=level):
                sf.write(self.reference, np.full(24000, level), 24000, subtype="FLOAT")
                output = self.root / "prepared.wav"
                with self.assertRaises(ValueError):
                    voice._prepare_reference(self.reference, output)
                self.assertFalse(output.exists())


class RealXTTSIntegration(unittest.TestCase):
    def test_real_model_reuses_voice_cache(self):
        if os.environ.get("NINECAST_RUN_XTTS_INTEGRATION") != "1":
            self.skipTest("Use --integration para executar XTTS real.")
        preferred = [config.VOICES_DIR / name for name in ("13seg.wav", "2.wav", "3.wav")]
        references = preferred if all(path.is_file() for path in preferred) else [
            path for path in sorted(config.VOICES_DIR.glob("*.wav"), key=lambda path: path.name)
            if path.is_file() and not path.name.startswith(".")
        ][:3]
        if not references:
            self.skipTest("As referências autorizadas de teste não estão disponíveis.")
        with tempfile.TemporaryDirectory(prefix="ninecast_integration_") as temporary:
            root = Path(temporary)
            with mock.patch.object(config, "VOICES_DIR", root):
                voice.get_or_create_voice_latents("teste_integracao", references)
                self.assertIn("teste_integracao", voice.list_available_voices())
                with mock.patch.object(voice, "_prepare_reference", side_effect=AssertionError("reprocessou")), \
                     mock.patch.object(voice._get_model(), "get_conditioning_latents", side_effect=AssertionError("extraiu novamente")):
                    for i in range(2):
                        metrics = voice.synthesize_speech(
                            "Olá, este é um teste de voz do NineCast.", "teste_integracao", root / f"{i}.wav"
                        )
                        self.assertGreater(metrics["audio_duration"], 0)
                        self.assertGreater(metrics["rtf"], 0)
                        print(f"Síntese {i + 1}: {metrics}", flush=True)
                self.assertEqual(sf.info(root / "1.wav").samplerate, 24000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--integration", action="store_true", help="Executa também o XTTS real em CPU.")
    parser.add_argument("--agree-license", action="store_true", help="Aceita os termos não comerciais do modelo.")
    options, unittest_args = parser.parse_known_args()
    if options.integration:
        if not options.agree_license and os.environ.get("COQUI_TOS_AGREED") != "1":
            parser.error("Para integração, aceite a licença com --agree-license.")
        os.environ["COQUI_TOS_AGREED"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["NINECAST_RUN_XTTS_INTEGRATION"] = "1"
    unittest.main(argv=[sys.argv[0], *unittest_args], verbosity=2)
