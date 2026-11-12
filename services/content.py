"""Extração local de texto de documentos PDF."""

import pymupdf


def extract_pdf_text(data: bytes) -> str:
    """Extrai o texto das páginas em ordem; PDFs digitalizados exigem OCR externo."""
    if not data:
        raise ValueError("O PDF está vazio.")
    try:
        with pymupdf.open(stream=data, filetype="pdf") as document:
            if document.needs_pass:
                raise ValueError("O PDF está protegido por senha.")
            pages = [page.get_text(sort=True).strip() for page in document]
    except pymupdf.FileDataError as exc:
        raise ValueError("O PDF é inválido ou está corrompido.") from exc
    text = "\n\n".join(page for page in pages if page)
    if not text:
        raise ValueError("Nenhum texto encontrado. PDFs digitalizados precisam de OCR, ainda não disponível.")
    return text
