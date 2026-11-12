"""Testes da extração local de PDFs."""

import unittest
import pymupdf

from services.content import extract_pdf_text


class ContentTests(unittest.TestCase):
    def test_multiple_pages(self):
        with pymupdf.open() as doc:
            for text in ("Primeira pagina", "Segunda pagina"):
                page = doc.new_page()
                page.insert_text((72, 72), text)
            data = doc.tobytes()
        self.assertEqual(extract_pdf_text(data), "Primeira pagina\n\nSegunda pagina")

    def test_invalid_empty_and_scanned(self):
        for data in (b"", b"invalido"):
            with self.assertRaises(ValueError):
                extract_pdf_text(data)
        with pymupdf.open() as doc:
            doc.new_page()
            data = doc.tobytes()
        with self.assertRaisesRegex(ValueError, "OCR"):
            extract_pdf_text(data)
