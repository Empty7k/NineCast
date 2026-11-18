"""Testes da interface Streamlit sem carregar o modelo de voz."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest

import config
from services import audio, library, voice


class AppTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        patcher = mock.patch.object(config, "PODCASTS_DIR", self.root / "podcasts")
        patcher.start()
        self.addCleanup(patcher.stop)

    def app(self):
        return AppTest.from_file(str(ROOT / "app.py"), default_timeout=20)

    def test_library_empty_and_populated(self):
        with mock.patch.object(library, "list_episodes", return_value=[]):
            app = self.app().run()
            self.assertFalse(app.exception)
            self.assertTrue(any("Nenhum episódio gravado" in item.value for item in app.info))
        path = self.root / "episodio.mp3"
        path.write_bytes(b"ID3 teste")
        episode = {"filename": path.name, "path": path, "size_mb": 1.0,
                   "created_at": "12/11/2026 14:30", "duration": "01:23"}
        with mock.patch.object(library, "list_episodes", return_value=[episode]):
            app = self.app().run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.get("audio")), 1)
            self.assertEqual(len(app.get("download_button")), 1)
            self.assertTrue(any("Total de episódios: 1" in item.value for item in app.caption))
            self.assertTrue(any("01:23" in item.value for item in app.caption))

    def test_library_delete_requires_confirmation(self):
        folder = config.PODCASTS_DIR
        folder.mkdir()
        path = folder / "episodio.mp3"
        path.write_bytes(b"ID3 teste")
        app = self.app().run()
        button = next(b for b in app.button if b.label == "Excluir episódio")
        self.assertTrue(button.disabled)
        next(c for c in app.checkbox if c.label == "Confirmar exclusão permanente").check().run()
        next(b for b in app.button if b.label == "Excluir episódio").click().run()
        self.assertFalse(app.exception)
        self.assertFalse(path.exists())
        self.assertTrue(any("excluído" in item.value for item in app.success))

    def test_library_loads_only_selected_episode(self):
        folder = config.PODCASTS_DIR
        folder.mkdir()
        (folder / "a.mp3").write_bytes(b"ID3 primeiro")
        (folder / "b.mp3").write_bytes(b"ID3 segundo")
        app = self.app().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.get("audio")), 1)
        self.assertEqual(len(app.get("download_button")), 1)
        selector = next(s for s in app.selectbox if s.label == "Escolha um episódio")
        selector.select("a.mp3").run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.get("audio")), 1)

    def test_import_pdf_updates_editable_script(self):
        fake_pdf = mock.Mock()
        fake_pdf.getvalue.return_value = b"pdf"
        def upload(label, **kwargs):
            return fake_pdf if label == "Documento PDF" else None
        with mock.patch.object(voice, "list_available_voices", return_value=[]), \
             mock.patch("streamlit.file_uploader", side_effect=upload), \
             mock.patch("services.content.extract_pdf_text", return_value="Texto extraído"):
            app = self.app().run()
            app.text_area[0].input("Roteiro anterior").run()
            next(b for b in app.button if b.label == "Extrair texto do PDF").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.text_area[0].value, "Texto extraído")
            app.text_area[0].input("Texto revisado").run()
            self.assertEqual(app.text_area[0].value, "Texto revisado")

    def test_render_with_and_without_voices(self):
        with mock.patch.object(voice, "list_available_voices", return_value=[]):
            app = self.app().run()
            self.assertFalse(app.exception)
            self.assertTrue(any("Nenhuma voz cadastrada" in item.value for item in app.info))
            self.assertEqual(len(app.selectbox), 0)
        with mock.patch.object(voice, "list_available_voices", return_value=["voz A", "voz B"]):
            app = self.app().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.selectbox[0].options, ["voz A", "voz B"])

    def test_script_metrics_and_chunk_estimate(self):
        script = "Olá, este é um roteiro de teste.\nAqui está o segundo parágrafo."
        with mock.patch.object(voice, "list_available_voices", return_value=["voz"]):
            app = self.app().run()
            app.text_area[0].input(script).run()
            self.assertFalse(app.exception)
            metrics = {item.label: item.value for item in app.metric}
            self.assertEqual(metrics["Palavras"], str(len(script.split())))
            self.assertEqual(metrics["Blocos XTTS"], "2")
            self.assertEqual(metrics["Duração estimada"], f"~{len(script.split()) / 150:.1f} min")
            expanding = " ".join(["R$ 123.456,78"] * 6)
            app.text_area[0].input(expanding).run()
            metrics = {item.label: item.value for item in app.metric}
            self.assertEqual(
                metrics["Blocos XTTS"],
                str(len(audio.split_text_into_chunks(audio.normalize_text_for_tts(expanding)))),
            )

    def test_generate_and_keep_result_after_interaction(self):
        podcast = self.root / "teste.mp3"
        podcast.write_bytes(b"ID3 amostra de MP3 para testar a interface")
        result = {"output_path": str(podcast), "audio_duration": 7.5, "total_seconds": 20.0, "rtf": 2.5}

        def generate(text, voice_name, progress_callback, episode_title=None):
            self.assertEqual(voice_name, "voz")
            self.assertEqual(episode_title, "Aula de testes")
            progress_callback(1, 1)
            return result

        with mock.patch.object(voice, "list_available_voices", return_value=["voz"]), \
             mock.patch.object(audio, "generate_podcast", side_effect=generate) as generate_mock:
            app = self.app().run()
            next(c for c in app.checkbox if c.label.startswith("Li e aceito")).check()
            next(item for item in app.text_input if item.label.startswith("Título do episódio")).input("Aula de testes")
            app.text_area[0].input("Este é um roteiro de teste para gerar o podcast.")
            next(button for button in app.button if button.label == "🎙️ Gerar Podcast").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(generate_mock.call_count, 1)
            self.assertEqual(len(app.get("audio")), 1)
            self.assertEqual(len(app.get("download_button")), 1)
            metrics = {item.label: item.value for item in app.metric}
            self.assertEqual(metrics["Duração"], "7.5 s")
            self.assertEqual(metrics["Geração"], "20.0 s")
            self.assertEqual(metrics["RTF de síntese"], "2.50×")
            app.text_area[0].input("Texto alterado após gerar o episódio.").run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.get("audio")), 1)
            self.assertEqual(generate_mock.call_count, 1)

    def test_text_limit_warns_and_blocks_generation(self):
        with mock.patch.object(config, "MAX_TEXT_CHARS", 10), \
             mock.patch.object(voice, "list_available_voices", return_value=["voz"]), \
             mock.patch.object(audio, "generate_podcast") as generate:
            app = self.app().run()
            next(c for c in app.checkbox if c.label.startswith("Li e aceito")).check()
            app.text_area[0].input("roteiro longo demais").run()
            self.assertTrue(any("excede o limite" in item.value for item in app.warning))
            next(button for button in app.button if button.label == "🎙️ Gerar Podcast").click().run()
            self.assertTrue(any("excede o limite" in item.value for item in app.error))
            generate.assert_not_called()

    def test_validation_and_friendly_memory_error(self):
        with mock.patch.object(voice, "list_available_voices", return_value=[]):
            app = self.app().run()
            next(button for button in app.button if button.label == "🎙️ Gerar Podcast").click().run()
            self.assertTrue(any("Escreva um texto" in item.value for item in app.error))
        with mock.patch.object(voice, "list_available_voices", return_value=["voz"]), \
             mock.patch.object(audio, "generate_podcast", side_effect=MemoryError):
            app = self.app().run()
            next(c for c in app.checkbox if c.label.startswith("Li e aceito")).check()
            app.text_area[0].input("Texto para gerar áudio.")
            next(button for button in app.button if button.label == "🎙️ Gerar Podcast").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(any("Memória insuficiente" in item.value for item in app.error))

    def test_license_required_for_generation_and_voice_registration(self):
        upload = mock.Mock()
        upload.getvalue.return_value = b"audio"
        with mock.patch.object(voice, "list_available_voices", return_value=["voz"]), \
             mock.patch.object(voice, "get_or_create_voice_latents") as create, \
             mock.patch.object(audio, "generate_podcast") as generate, \
             mock.patch("streamlit.file_uploader", return_value=[upload]):
            app = self.app().run()
            next(item for item in app.text_input if item.label == "Nome da voz").input("outra voz")
            next(b for b in app.button if b.label == "Cadastrar voz").click().run()
            self.assertTrue(any("aceite a licença" in e.value for e in app.error))
            create.assert_not_called()
            app.text_area[0].input("Texto de exemplo")
            next(b for b in app.button if b.label == "🎙️ Gerar Podcast").click().run()
            self.assertTrue(any("aceite a licença" in e.value for e in app.error))
            generate.assert_not_called()

    def test_add_voice_upload_is_temporary(self):
        for count in (1, 2, 5):
            with self.subTest(samples=count):
                names, reference_paths = [], []
                contents = [f"WAV {index}".encode() for index in range(count)]
                fake_uploads = [mock.Mock(name="mesmo_nome.wav") for _ in contents]
                for upload, data in zip(fake_uploads, contents):
                    upload.getvalue.return_value = data

                def upload_files(label, **kwargs):
                    if label == "Documento PDF":
                        return None
                    self.assertTrue(kwargs["accept_multiple_files"])
                    self.assertEqual(kwargs["type"], ["wav"])
                    return fake_uploads

                def create(name, paths):
                    # Todas as amostras existem antes da extração, na ordem do upload.
                    self.assertEqual([path.read_bytes() for path in paths], contents)
                    self.assertEqual(len(set(paths)), count)
                    reference_paths.extend(paths)
                    names.append(name)

                with mock.patch.object(config, "VOICES_DIR", self.root / "voices"), \
                     mock.patch.object(voice, "list_available_voices", side_effect=lambda: list(names)), \
                     mock.patch.object(voice, "get_or_create_voice_latents", side_effect=create), \
                     mock.patch("streamlit.file_uploader", side_effect=upload_files):
                    app = self.app().run()
                    next(c for c in app.checkbox if c.label.startswith("Li e aceito")).check()
                    next(item for item in app.text_input if item.label == "Nome da voz").input("Nova voz")
                    next(button for button in app.button if button.label == "Cadastrar voz").click().run()
                    self.assertFalse(app.exception)
                    self.assertEqual(app.selectbox[0].value, "Nova voz")
                    self.assertTrue(any("cadastrada com sucesso" in item.value for item in app.success))
                    self.assertEqual(list((self.root / "voices").iterdir()), [])
                    self.assertTrue(all(not path.exists() for path in reference_paths))

    def test_voice_registration_rejects_more_than_five_samples(self):
        uploads = [mock.Mock() for _ in range(6)]
        with mock.patch.object(voice, "list_available_voices", return_value=[]), \
             mock.patch.object(voice, "get_or_create_voice_latents") as create, \
             mock.patch("streamlit.file_uploader", return_value=uploads):
            app = self.app().run()
            next(item for item in app.text_input if item.label == "Nome da voz").input("Nova voz")
            next(button for button in app.button if button.label == "Cadastrar voz").click().run()
            self.assertTrue(any("no máximo 5" in item.value for item in app.error))
            create.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
