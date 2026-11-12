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
        episode = {"filename": path.name, "path": path, "size_mb": 1.0, "created_at": "12/11/2026 14:30"}
        with mock.patch.object(library, "list_episodes", return_value=[episode]):
            app = self.app().run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.get("audio")), 1)
            self.assertEqual(len(app.get("download_button")), 1)
            self.assertTrue(any("Total de episódios: 1" in item.value for item in app.markdown))

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
            expected = f"{len(script.split())} palavras · {len(script)} caracteres · Estimativa de ~2 blocos"
            self.assertTrue(any(expected in item.value for item in app.caption))

    def test_generate_and_keep_result_after_interaction(self):
        podcast = self.root / "teste.mp3"
        podcast.write_bytes(b"ID3 amostra de MP3 para testar a interface")
        result = {"output_path": str(podcast), "audio_duration": 7.5, "total_seconds": 20.0, "rtf": 2.5}

        def generate(text, voice_name, progress_callback):
            self.assertEqual(voice_name, "voz")
            progress_callback(1, 1)
            return result

        with mock.patch.object(voice, "list_available_voices", return_value=["voz"]), \
             mock.patch.object(audio, "generate_podcast", side_effect=generate) as generate_mock:
            app = self.app().run()
            next(c for c in app.checkbox if c.label.startswith("Li e aceito")).check()
            app.text_area[0].input("Este é um roteiro de teste para gerar o podcast.")
            next(button for button in app.button if button.label == "🎙️ Gerar Podcast").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(generate_mock.call_count, 1)
            self.assertEqual(len(app.get("audio")), 1)
            self.assertEqual(len(app.get("download_button")), 1)
            self.assertTrue(any("Duração: 7.5 s" in item.value and "RTF de síntese: 2.50×" in item.value
                                for item in app.caption))
            app.text_area[0].input("Texto alterado após gerar o episódio.").run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.get("audio")), 1)
            self.assertEqual(generate_mock.call_count, 1)

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

    def test_add_voice_upload_is_temporary(self):
        names = []
        fake_upload = mock.Mock()
        fake_upload.getvalue.return_value = b"WAV simulado"
        def create(name, paths):
            self.assertEqual(paths[0].read_bytes(), b"WAV simulado")
            names.append(name)
        with mock.patch.object(config, "VOICES_DIR", self.root / "voices"), \
             mock.patch.object(voice, "list_available_voices", side_effect=lambda: list(names)), \
             mock.patch.object(voice, "get_or_create_voice_latents", side_effect=create), \
             mock.patch("streamlit.file_uploader", return_value=fake_upload):
            app = self.app().run()
            next(c for c in app.checkbox if c.label.startswith("Li e aceito")).check()
            app.text_input[0].input("Nova voz")
            next(button for button in app.button if button.label == "Cadastrar voz").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.selectbox[0].value, "Nova voz")
            self.assertTrue(any("cadastrada com sucesso" in item.value for item in app.success))
            self.assertEqual(list((self.root / "voices").iterdir()), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
