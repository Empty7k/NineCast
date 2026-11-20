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
    t = np.arange(rate // 10) / rate
    samples = 0.15 * np.cos(2 * np.pi * 200 * t)
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

    def test_normalize_text_for_tts(self):
        raw = (
            "• § 2º e 1ª regra: em 17/11/2026 ou 05/03, taxa de 15% e 3,5%.\n"
            "---\n"
            "Valor de R$ 150,00, R$ 1,50, 3,14 e 25.000 itens em P&D (código 45/99)."
        )
        normalized = audio.normalize_text_for_tts(raw)
        self.assertIn("parágrafo segundo e primeira regra", normalized)
        self.assertIn("dezessete de novembro de dois mil e vinte e seis", normalized)
        self.assertIn("cinco de março", normalized)
        self.assertIn("quinze por cento e três vírgula cinco por cento", normalized)
        self.assertIn("cento e cinquenta reais", normalized)
        self.assertIn("um real e cinquenta centavos", normalized)
        self.assertIn("três vírgula quatorze", normalized)
        self.assertIn("vinte e cinco mil itens em P e D", normalized)
        self.assertIn("código 45/99", normalized)
        self.assertNotIn("•", normalized)
        self.assertNotIn("---", normalized)

    def test_normalization_regressions(self):
        cases = {
            "3.14; 0,05!": "três vírgula quatorze; zero vírgula zero cinco!",
            "Versão 3.12.1, IP 192.168.0.1, ISO-9001 e ABC123":
                "Versão 3.12.1, IP 192.168.0.1, ISO-9001 e ABC123",
            "31/04/2026 e 29/02/2025": "31/04/2026 e 29/02/2025",
            "29/02/2024": "vinte e nove de fevereiro de dois mil e vinte e quatro",
            "01/02/2026": "primeiro de fevereiro de dois mil e vinte e seis",
            "R$ 0,05; R$ 1,00": "cinco centavos; um real",
            "R$ 1.000.000,00": "um milhão de reais",
            "R$ 2.000.000.000,00": "dois bilhões de reais",
            "-5 e -3.5%": "menos cinco e menos três vírgula cinco por cento",
            "§ 2º; §§ 3º e 4º; 2ª & 10º":
                "parágrafo segundo; parágrafos terceiro e quarto; segunda e décimo",
            "0º e 1000º": "0º e 1000º",
            "001 e 1000000000000": "001 e 1000000000000",
            "1.000.000.001": "um bilhão e um",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(audio.normalize_text_for_tts(raw), expected)

    def test_normalization_preserves_paragraphs(self):
        self.assertEqual(audio.normalize_text_for_tts("• 2\n\n---\n▪ 3"),
                         "dois\n\n\ntrês")


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
        for start, end in ((0, .1), (.4, .5), (1.1, 1.2)):
            self.assertEqual(samples[round(start * rate)], 0)
            self.assertEqual(samples[round(end * rate) - 1], 0)
            self.assertGreater(abs(samples[round((start + .05) * rate)]), 0)

    def test_generate_rejects_text_above_configured_limit(self):
        with mock.patch.object(config, "MAX_TEXT_CHARS", 10), \
             mock.patch.object(audio.voice, "synthesize_speech") as synthesize:
            with self.assertRaisesRegex(ValueError, "limite de 10"):
                audio.generate_podcast("texto longo demais", "voz", self.root / "saida.mp3")
        synthesize.assert_not_called()

    def test_trim_keeps_speech_margins_internal_pauses_and_configured_gaps(self):
        rate = 24000
        # Duas falas de 0,1 s com uma pausa interna de 0,2 s, que deve permanecer.
        tone = 0.2 * np.cos(2 * np.pi * 200 * np.arange(rate // 10) / rate)
        speech = np.concatenate([tone, np.zeros(rate // 5), tone])
        paths = []
        for index, (leading, trailing) in enumerate(((.2, .4), (.35, .2), (.15, .3))):
            path = self.root / f"padded_{index}.wav"
            sf.write(path, np.concatenate([np.zeros(round(leading * rate)), speech,
                                          np.zeros(round(trailing * rate))]), rate, subtype="PCM_16")
            paths.append(path)
        output = self.root / "trimmed.wav"
        duration = audio._concatenate_wavs(paths, [0, 0, 1], output)
        samples, _ = sf.read(output)
        block_frames = len(speech) + round(.08 * rate)
        self.assertAlmostEqual(duration, 3 * block_frames / rate + .3 + .6, places=6)
        self.assertEqual(len(samples), round(duration * rate))
        first_end = block_frames
        second_start = first_end + round(.3 * rate)
        second_end = second_start + block_frames
        self.assertTrue(np.all(samples[first_end:second_start] == 0))
        self.assertTrue(np.all(samples[second_end:second_end + round(.6 * rate)] == 0))
        # As margens de 40 ms permanecem; nenhuma fala ou pausa interna foi cortada.
        margin = round(.04 * rate)
        self.assertTrue(np.all(samples[:margin] == 0))
        self.assertTrue(np.all(samples[round(.14 * rate):round(.34 * rate)] == 0))
        self.assertGreater(abs(samples[margin + round(.05 * rate)]), .01)
        self.assertGreater(abs(samples[margin + round(.35 * rate)]), .01)

    def test_rms_leveling_reduces_volume_gap_without_clipping(self):
        rate = 24000
        paths = []
        for index, amplitude in enumerate((.02, .2)):
            path = self.root / f"level_{index}.wav"
            sf.write(path, amplitude * np.cos(2 * np.pi * 200 * np.arange(rate) / rate),
                     rate, subtype="PCM_16")
            paths.append(path)
        output = self.root / "leveled.wav"
        self.assertAlmostEqual(audio._concatenate_wavs(paths, [0, 0], output), 2.3)
        samples, _ = sf.read(output)
        rms_quiet = np.sqrt(np.mean(samples[2400:21600] ** 2))
        rms_loud = np.sqrt(np.mean(samples[33600:52800] ** 2))
        self.assertLess(rms_loud / rms_quiet, 5)  # Antes: razão 10 (20 dB).
        self.assertLessEqual(rms_quiet / (.02 / np.sqrt(2)), 2.01)  # Ganho <= 6 dB.
        self.assertLessEqual(np.max(np.abs(samples)), 10 ** (-1 / 20) + 1 / 32768)

    def test_peak_limit_preserves_transient_without_integer_overflow(self):
        rate = 24000
        quiet = .02 * np.cos(2 * np.pi * 200 * np.arange(rate) / rate)
        quiet[rate // 2] = -1.0
        paths = [self.root / "transient.wav", self.root / "loud.wav"]
        sf.write(paths[0], quiet, rate, subtype="PCM_16")
        sf.write(paths[1], .3 * np.cos(2 * np.pi * 200 * np.arange(rate) / rate),
                 rate, subtype="PCM_16")
        output = self.root / "limited.wav"
        audio._concatenate_wavs(paths, [0, 0], output)
        samples, _ = sf.read(output)
        self.assertAlmostEqual(samples[rate // 2], -10 ** (-1 / 20), places=4)
        self.assertLess(np.max(np.abs(samples)), .892)
        # O bloco recebe um ganho constante; não há achatamento do transiente.
        original, _ = sf.read(paths[0])
        np.testing.assert_allclose(samples[2400:21600], original[2400:21600] * 10 ** (-1 / 20),
                                   atol=1 / 32768)

    def test_silent_and_very_short_blocks_are_safe(self):
        paths = [self.root / "silent.wav", self.root / "one_sample.wav"]
        sf.write(paths[0], np.zeros(4800), 24000, subtype="PCM_16")
        sf.write(paths[1], [-1.0], 24000, subtype="PCM_16")
        output = self.root / "short.wav"
        duration = audio._concatenate_wavs(paths, [0, 0], output)
        samples, _ = sf.read(output)
        self.assertAlmostEqual(duration, .5 + 1 / 24000)
        self.assertTrue(np.all(samples[:-1] == 0))
        self.assertLess(abs(samples[-1]), .892)

    def test_fades_across_streaming_buffers_preserve_duration_and_middle(self):
        rate = 24000
        source = self.root / "long.wav"
        sf.write(source, np.full(round(2.3 * rate), .15), rate, subtype="PCM_16")
        output = self.root / "long_faded.wav"
        duration = audio._concatenate_wavs([source], [0], output)
        samples, _ = sf.read(output)
        original, _ = sf.read(source)
        self.assertAlmostEqual(duration, 2.3)
        self.assertEqual(len(samples), len(original))
        self.assertEqual(samples[0], 0)
        self.assertEqual(samples[-1], 0)
        # Só as bordas têm rampa, incluindo quando o fim cai em outro buffer.
        np.testing.assert_array_equal(samples[480:-480], original[480:-480])
        self.assertTrue(np.all(np.diff(samples[:240]) >= 0))
        self.assertTrue(np.all(np.diff(samples[-240:]) <= 0))

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg/ffprobe indisponível")
    def test_real_ffmpeg_and_progress(self):
        text = "Primeiro parágrafo com conteúdo para teste.\nSegundo parágrafo para validar a montagem."
        output = self.root / "episódio teste.mp3"
        updates = []
        with mock.patch.object(audio.voice, "synthesize_speech", side_effect=fake_synthesis), \
             mock.patch.object(audio.subprocess, "run", wraps=subprocess.run) as encode:
            result = audio.generate_podcast(text, "voz", output, lambda n, total: updates.append((n, total)))
        command = encode.call_args.args[0]
        chain = command[command.index("-af") + 1]
        self.assertEqual([item.split("=")[0] for item in chain.split(",")], ["highpass", "acompressor", "loudnorm"])
        self.assertIn("highpass=f=80", chain)
        self.assertIn("loudnorm=I=-16:TP=-1.5:LRA=9", chain)
        self.assertEqual(updates, [(1, 2), (2, 2)])
        self.assertEqual(result["chunk_count"], 2)
        self.assertAlmostEqual(result["audio_duration"], .8, places=3)
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                                "stream=codec_name,sample_rate,channels,bit_rate",
                                "-of", "json", str(output)], capture_output=True, text=True, check=True)
        stream = json.loads(probe.stdout)["streams"][0]
        self.assertEqual((stream["codec_name"], stream["sample_rate"], stream["channels"], stream["bit_rate"]),
                         ("mp3", "48000", 1, "192000"))
        decoded = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(output), "-f", "f32le", "pipe:1"],
            capture_output=True, check=True,
        )
        samples = np.frombuffer(decoded.stdout, dtype="<f4")
        self.assertTrue(np.isfinite(samples).all())
        self.assertGreater(np.max(np.abs(samples)), .001)
        self.assertLess(np.max(np.abs(samples)), 1.0)
        self.assertEqual(list(self.root.glob(".ninecast_audio_*")), [])

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg/ffprobe indisponível")
    def test_qwen_batched_pipeline_normalizes_and_exports_id3(self):
        text = "Reserve 15% da memória para o cache.\nConfira as métricas durante o uso."
        output = self.root / "qwen.mp3"
        updates = []

        def batch(chunks, voice_name, paths, callback):
            self.assertEqual(voice_name, "Voz Qwen")
            self.assertEqual(" ".join(chunks), "Reserve quinze por cento da memória para o cache. "
                                              "Confira as métricas durante o uso.")
            self.assertTrue(all(len(chunk) <= 35 for chunk in chunks))
            results = []
            for index, (chunk, path) in enumerate(zip(chunks, paths), 1):
                results.append(fake_synthesis(chunk, voice_name, path))
                callback(index, len(chunks))
            return results

        with mock.patch.object(config, "QWEN_TTS_MAX_CHARS", 35), \
             mock.patch.object(audio.qwen_tts, "synthesize_chunks", side_effect=batch) as synthesize, \
             mock.patch.object(voice, "synthesize_speech") as xtts:
            result = audio.generate_podcast(
                text, "Voz Qwen", output, lambda n, total: updates.append((n, total)),
                episode_title="Memória", engine="qwen3_06b",
            )
        synthesize.assert_called_once()
        xtts.assert_not_called()
        self.assertEqual(updates, [(1, 3), (2, 3), (3, 3)])
        self.assertEqual(result["chunk_count"], 3)
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                                "stream=bit_rate,sample_rate:format_tags=title,artist", "-of", "json",
                                str(output)], capture_output=True, text=True, check=True)
        tags = json.loads(probe.stdout)
        self.assertEqual(tags["format"]["tags"], {"title": "Memória", "artist": "Voz Qwen"})
        self.assertEqual(tags["streams"][0], {"sample_rate": "48000", "bit_rate": "192000"})

    def test_unknown_engine_rejected_before_synthesis(self):
        with mock.patch.object(audio.qwen_tts, "synthesize_chunks") as synthesize:
            with self.assertRaisesRegex(ValueError, "motor de voz"):
                audio.generate_podcast("Roteiro.", "Voz", engine="desconhecido")
            synthesize.assert_not_called()

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg/ffprobe indisponível")
    def test_title_creates_unique_sanitized_name_and_id3_metadata(self):
        existing = self.root / "algebra-cafe-aaaaaaaaaa.mp3"
        existing.write_bytes(b"nao sobrescrever")
        ids = [mock.Mock(hex="aaaaaaaaaa000000"), mock.Mock(hex="bbbbbbbbbb000000")]
        with mock.patch.object(config, "PODCASTS_DIR", self.root), \
             mock.patch.object(audio.uuid, "uuid4", side_effect=ids), \
             mock.patch.object(audio.voice, "synthesize_speech", side_effect=fake_synthesis):
            result = audio.generate_podcast("Conteúdo para narrar.", "Voz de teste",
                                            episode_title="Álgebra & Café")
        output = Path(result["output_path"])
        self.assertEqual(output.name, "algebra-cafe-bbbbbbbbbb.mp3")
        self.assertEqual(existing.read_bytes(), b"nao sobrescrever")
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format_tags=title,artist",
             "-of", "json", str(output)], capture_output=True, text=True, check=True
        )
        tags = json.loads(probe.stdout)["format"]["tags"]
        self.assertEqual(tags["title"], "Álgebra & Café")
        self.assertEqual(tags["artist"], "Voz de teste")

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

    def test_generate_podcast_normalizes_text_before_synthesis(self):
        seen_chunks = []

        def capture_synthesis(text, voice_name, output_path):
            seen_chunks.append(text)
            return fake_synthesis(text, voice_name, output_path)

        output = self.root / "normalizado.mp3"
        with mock.patch.object(voice, "synthesize_speech", side_effect=capture_synthesis):
            audio.generate_podcast("Meta de 15% em 05/03 por R$ 10,00.", "voz", output)
        self.assertEqual(len(seen_chunks), 1)
        self.assertIn("quinze por cento", seen_chunks[0])
        self.assertIn("cinco de março", seen_chunks[0])
        self.assertIn("dez reais", seen_chunks[0])

    def test_normalization_expansion_is_chunked_after_original_limit_check(self):
        raw = "2026 " * 12
        seen = []

        def capture(text, voice_name, output_path):
            seen.append(text)
            return fake_synthesis(text, voice_name, output_path)

        with mock.patch.object(config, "MAX_TEXT_CHARS", len(raw)), \
             mock.patch.object(config, "XTTS_MAX_CHARS", 60), \
             mock.patch.object(voice, "synthesize_speech", side_effect=capture):
            result = audio.generate_podcast(raw, "voz", self.root / "expandido.mp3")
        self.assertGreater(len(seen), 1)
        self.assertTrue(all(len(chunk) <= 60 for chunk in seen))
        self.assertEqual(" ".join(seen), audio.normalize_text_for_tts(raw).strip())
        self.assertEqual(result["chunk_count"], len(seen))


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
