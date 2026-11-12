"""Testes do fatiador e do FFmpeg; integração XTTS real é opcional."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config
from services import audio, voice


def fake_synthesis(text, voice_name, output_path):
    rate = 24000
    samples = np.full(rate // 10, 0.15, dtype=np.float32)
    sf.write(output_path, samples, rate, subtype="PCM_16")
    return {"synthesis_seconds": 0.05, "audio_duration": 0.1,
            "rtf": 0.5, "output_path": str(output_path)}


class TextChunkTests(unittest.TestCase):
    def test_paragraphs_and_short_greeting(self):
        text = "Olá! Este é um primeiro parágrafo completo.\n\nOutro parágrafo contém mais informações."
        chunks = audio.split_text_into_chunks(text, 80)
        self.assertEqual(chunks, ["Olá! Este é um primeiro parágrafo completo.",
                                  "Outro parágrafo contém mais informações."])

    def test_abbreviations_decimals_and_complex_punctuation(self):
        text = "O Dr. Silva explicou a versão 3.14. Depois, por ex. demonstrou: leitura, escrita e teste!"
        chunks = audio.split_text_into_chunks(text, 50)
        self.assertEqual(chunks[0], "O Dr. Silva explicou a versão 3.14.")
        self.assertEqual(" ".join(chunks), text)
        self.assertTrue(all(len(chunk) <= 50 for chunk in chunks))

    def test_long_sentence_and_unbroken_word(self):
        text = "Aula sobre algoritmos: " + "palavra " * 35 + "fim."
        chunks = audio.split_text_into_chunks(text, 60)
        self.assertTrue(all(20 <= len(chunk) <= 60 for chunk in chunks), chunks)
        self.assertEqual(" ".join(chunks), " ".join(text.split()))
        chunks = audio.split_text_into_chunks("x" * 85, 30)
        self.assertTrue(all(len(c) <= 30 for c in chunks))
        self.assertEqual("".join(chunks), "x" * 85)

    def test_input_validation(self):
        for value in ("", " \n ", None):
            with self.assertRaises(ValueError):
                audio.split_text_into_chunks(value)
        for limit in (0, 19, True, 3.5):
            with self.assertRaises(ValueError):
                audio.split_text_into_chunks("Algum texto", limit)


class AudioPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_concatenate_silence_between_sentences_and_paragraphs(self):
        paths = []
        for i in range(3):
            path = self.root / f"{i}.wav"
            fake_synthesis("texto", "voz", path)
            paths.append(path)
        combined = self.root / "combined.wav"
        duration = audio._concatenate_wavs(paths, [0, 0, 1], combined)
        self.assertAlmostEqual(duration, 1.2, places=3)
        samples, rate = sf.read(combined)
        self.assertEqual(rate, 24000)
        self.assertTrue(np.all(samples[round(.1 * rate):round(.4 * rate)] == 0))
        self.assertTrue(np.all(samples[round(.5 * rate):round(1.1 * rate)] == 0))

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg/ffprobe indisponível")
    def test_real_ffmpeg_and_progress(self):
        text = "Primeiro parágrafo com conteúdo para teste.\nSegundo parágrafo para validar a montagem."
        output = self.root / "episódio teste.mp3"
        updates = []
        with mock.patch.object(audio.voice, "synthesize_speech", side_effect=fake_synthesis):
            result = audio.generate_podcast(text, "voz", output, lambda n, total: updates.append((n, total)))
        self.assertEqual(updates, [(1, 2), (2, 2)])
        self.assertEqual(result["chunk_count"], 2)
        self.assertAlmostEqual(result["audio_duration"], .8, places=3)
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                                "stream=codec_name,sample_rate,channels,bit_rate",
                                "-of", "json", str(output)], capture_output=True, text=True, check=True)
        stream = json.loads(probe.stdout)["streams"][0]
        self.assertEqual((stream["codec_name"], stream["sample_rate"], stream["channels"], stream["bit_rate"]),
                         ("mp3", "24000", 1, "128000"))
        self.assertEqual(list(self.root.glob(".ninecast_audio_*")), [])

    def test_cleanup_after_callback_and_ffmpeg_failure(self):
        output = self.root / "output.mp3"
        def fail_progress(*args):
            raise RuntimeError("interrompido")
        with mock.patch.object(audio.voice, "synthesize_speech", side_effect=fake_synthesis):
            with self.assertRaisesRegex(RuntimeError, "interrompido"):
                audio.generate_podcast("Primeiro parágrafo.", "voz", output, fail_progress)
            with mock.patch.object(audio.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "ffmpeg", stderr="erro")):
                with self.assertRaisesRegex(RuntimeError, "erro"):
                    audio.generate_podcast("Primeiro parágrafo.", "voz", output)
        self.assertFalse(output.exists())
        self.assertEqual(list(self.root.glob(".ninecast_audio_*")), [])

    def test_cleanup_when_second_synthesis_fails(self):
        output = self.root / "output.mp3"
        with mock.patch.object(audio.voice, "synthesize_speech",
                               side_effect=[mock.DEFAULT, RuntimeError("falha na voz")],
                               wraps=fake_synthesis):
            with self.assertRaisesRegex(RuntimeError, "falha na voz"):
                audio.generate_podcast("Primeiro parágrafo.\nSegundo parágrafo.", "voz", output)
        self.assertFalse(output.exists())
        self.assertEqual(list(self.root.glob(".ninecast_audio_*")), [])

    def test_refuses_overwrite_and_wrong_extension(self):
        path = self.root / "existing.mp3"
        path.write_bytes(b"original")
        with self.assertRaises(FileExistsError):
            audio.generate_podcast("Texto", "voz", path)
        with self.assertRaises(ValueError):
            audio.generate_podcast("Texto", "voz", self.root / "saida.wav")
        self.assertEqual(path.read_bytes(), b"original")

    def test_does_not_replace_file_created_during_generation(self):
        path = self.root / "existing.mp3"
        def synthesize_and_create(text, voice_name, output_path):
            path.write_bytes(b"criado por outro processo")
            return fake_synthesis(text, voice_name, output_path)
        with mock.patch.object(audio.voice, "synthesize_speech", side_effect=synthesize_and_create):
            with self.assertRaises(FileExistsError):
                audio.generate_podcast("Texto para o episodio.", "voz", path)
        self.assertEqual(path.read_bytes(), b"criado por outro processo")
        self.assertEqual(list(self.root.glob(".ninecast_audio_*")), [])

    def test_move_failure_cleans_reservation(self):
        path = self.root / "output.mp3"
        with mock.patch.object(audio.voice, "synthesize_speech", side_effect=fake_synthesis), \
             mock.patch.object(audio.shutil, "move", side_effect=OSError("falha ao mover")):
            with self.assertRaisesRegex(OSError, "falha ao mover"):
                audio.generate_podcast("Texto para o episodio.", "voz", path)
        self.assertFalse(path.exists())
        self.assertEqual(list(self.root.glob(".ninecast_audio_*")), [])


class RealIntegration(unittest.TestCase):
    def test_real_two_paragraph_podcast(self):
        if os.environ.get("NINECAST_AUDIO_INTEGRATION") != "1":
            self.skipTest("Use --integration para executar XTTS real.")
        preferred = [config.VOICES_DIR / name for name in ("13seg.wav", "2.wav", "3.wav")]
        references = preferred if all(path.is_file() for path in preferred) else [
            path for path in sorted(config.VOICES_DIR.glob("*.wav"), key=lambda path: path.name)
            if path.is_file() and not path.name.startswith(".")
        ][:3]
        if not references:
            self.skipTest("Referências autorizadas não encontradas.")
        with tempfile.TemporaryDirectory(prefix="ninecast_audio_integration_") as temp:
            root = Path(temp)
            with mock.patch.object(config, "VOICES_DIR", root):
                voice.get_or_create_voice_latents("teste", references)
                updates = []
                result = audio.generate_podcast(
                    "Olá, este é um teste de narração local.\nAgora apresentamos o segundo parágrafo do podcast.",
                    "teste", root / "episodio.mp3", lambda n, total: updates.append((n, total))
                )
                self.assertEqual(updates, [(1, 2), (2, 2)])
                self.assertEqual(result["chunk_count"], 2)
                self.assertGreater(result["audio_duration"], 0)
                self.assertGreater((root / "episodio.mp3").stat().st_size, 0)
                print(f"Métricas da integração: {result}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--integration", action="store_true")
    parser.add_argument("--agree-license", action="store_true")
    options, remaining = parser.parse_known_args()
    if options.integration:
        if not options.agree_license and os.environ.get("COQUI_TOS_AGREED") != "1":
            parser.error("Para integração, aceite a licença com --agree-license.")
        os.environ["COQUI_TOS_AGREED"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["NINECAST_AUDIO_INTEGRATION"] = "1"
    unittest.main(argv=[sys.argv[0], *remaining], verbosity=2)
