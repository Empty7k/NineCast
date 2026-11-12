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

    def get_conditioning_latents(self, **kwargs):
        self.extractions += 1
        return torch.ones(1, 2), torch.ones(1, 3)

    def inference(self, **kwargs):
        return {"wav": np.full(24000, 0.1, dtype=np.float32)}


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
        with mock.patch.object(voice, "_prepare_reference", side_effect=AssertionError("reprocessou")):
            for index in range(2):
                result = voice.synthesize_speech("Olá, este é um teste.", "teste", self.root / f"{index}.wav")
                self.assertEqual(result["audio_duration"], 1.0)
                self.assertGreaterEqual(result["rtf"], 0)
            voice.get_or_create_voice_latents("teste", [self.reference])
        self.assertEqual(self.fake.extractions, 1)
        self.assertEqual(sf.info(self.root / "1.wav").samplerate, 24000)

    def test_changed_reference_invalidates_cache(self):
        voice.get_or_create_voice_latents("teste", [self.reference])
        sf.write(self.reference, np.ones(24000, dtype=np.float32) * 0.1, 24000)
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
        sf.write(self.reference, np.full(rate * 3, 0.1, dtype=np.float32), rate)
        intervals = np.array([[0, 12000], [18000, 30000], [42000, 54000]])
        output = self.root / "prepared.wav"
        with mock.patch("librosa.effects.split", return_value=intervals):
            voice._prepare_reference(self.reference, output)
        prepared, _ = sf.read(output)
        # A pausa de 0,25 s é preservada; a de 0,5 s vira 0,15 s mais padding.
        self.assertAlmostEqual(len(prepared) / rate, 1.99, places=2)


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
