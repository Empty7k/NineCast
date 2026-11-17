"""Extração local de texto de documentos PDF."""

import re
from collections import Counter
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
    repeated = set()
    if len(pages) >= 3:
        edges = Counter()
        for page in pages:
            lines = [line.strip() for line in page.splitlines() if line.strip()]
            edges.update({line for line in (lines[:1] + lines[-1:]) if len(line) <= 100})
        repeated = {line for line, count in edges.items() if count >= max(3, len(pages) - 1)}
    cleaned = []
    for page in pages:
        lines = page.splitlines()
        nonempty = [i for i, line in enumerate(lines) if line.strip()]
        edges = {nonempty[0], nonempty[-1]} if nonempty else set()
        lines = [line for i, line in enumerate(lines)
                 if not (i in edges and line.strip() in repeated)
                 and not re.fullmatch(r'\s*(?:-\s*)?(?:(?:página|pág\.)\s*)?\d+(?:\s+de\s+\d+)?(?:\s*-)?\s*', line, re.IGNORECASE)]
        page = "\n".join(lines)
        page = re.sub(r'https?://[^\s<>]+|www\.[^\s<>]+', '', page, flags=re.IGNORECASE)
        cleaned.append(page.strip())
    text = "\n\n".join(page for page in cleaned if page)
    if not text:
        raise ValueError("Nenhum texto encontrado. PDFs digitalizados precisam de OCR, ainda não disponível.")
    # Apenas uma quebra de linha: não une palavras através de parágrafos.
    text = re.sub(r"(\b\w+)-[^\S\n\r]*\r?\n[^\S\n\r]*(\w+\b)", r"\1\2", text)
    return re.sub(r'\n[ \t]*\n(?:[ \t]*\n)+', '\n\n', text)
