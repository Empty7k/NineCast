"""Testes da extração local de PDFs."""

import unittest
import pymupdf

from services.content import extract_pdf_text


class ContentTests(unittest.TestCase):
    def test_hyphens_and_paragraphs(self):
        from unittest.mock import patch
        class Page:
            def get_text(self, **kwargs):
                return "desen- \n volvimento e infor-\nmação\n\npalavra-\n\nseguinte"
        class Document:
            needs_pass = False
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def __iter__(self): return iter([Page()])
        with patch("services.content.pymupdf.open", return_value=Document()):
            self.assertEqual(extract_pdf_text(b"pdf"), "desenvolvimento e informação\n\npalavra-\n\nseguinte")
    def test_multiple_pages(self):
        with pymupdf.open() as doc:
            for text in ("Primeira pagina", "Segunda pagina"):
                page = doc.new_page()
                page.insert_text((72, 72), text)
            data = doc.tobytes()
        self.assertEqual(extract_pdf_text(data), "Primeira pagina\n\nSegunda pagina")

    def test_actual_pdf_with_accents(self):
        with pymupdf.open() as doc:
            page = doc.new_page()
            page.insert_text((72, 72), "infor-\nmação")
            data = doc.tobytes()
        self.assertEqual(extract_pdf_text(data), "informação")

    def test_invalid_empty_and_scanned(self):
        for data in (b"", b"invalido"):
            with self.assertRaises(ValueError):
                extract_pdf_text(data)
        with pymupdf.open() as doc:
            doc.new_page()
            data = doc.tobytes()
        with self.assertRaisesRegex(ValueError, "OCR"):
            extract_pdf_text(data)
