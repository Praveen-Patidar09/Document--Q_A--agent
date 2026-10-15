"""Paragraph- and sentence-aware chunking that carries page/slide/sheet metadata forward."""

import re

from app.config import settings
from app.models import Chunk, DocumentPart

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?।॥])\s+(?=[A-Z0-9\u0900-\u097f])")
_HEADING = re.compile(r"^(?:#{1,6}\s+|.{1,100}:$)")


def _split_long_paragraph(text: str, size: int) -> list[str]:
    sentences = _SENTENCE_BOUNDARY.split(text)
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > size:
            if current:
                pieces.append(current)
                current = ""
            start = 0
            while start < len(sentence):
                end = min(len(sentence), start + size)
                if end < len(sentence):
                    space = sentence.rfind(" ", start + size // 2, end)
                    if space > start:
                        end = space
                pieces.append(sentence[start:end].strip())
                start = end
                while start < len(sentence) and sentence[start].isspace():
                    start += 1
            continue
        if current and len(current) + 1 + len(sentence) > size:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


def split_text(text: str, size: int = settings.chunk_size,
               overlap: int = settings.chunk_overlap) -> list[str]:
    if size < 1 or overlap < 0 or overlap >= size:
        raise ValueError("Chunk size must be positive and overlap must be in [0, size).")
    paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n", text) if p.strip()]
    atomic: list[str] = []
    for paragraph in paragraphs:
        atomic.extend(_split_long_paragraph(paragraph, size))
    if not atomic:
        return []

    chunks: list[str] = []
    current = ""
    for part in atomic:
        if current and len(current) + 2 + len(part) > size:
            chunks.append(current)
            # Keep overlap only at a complete word boundary; it preserves local context without
            # merging distant paragraphs into a single fragment.
            words = current.split()
            carry: list[str] = []
            carry_len = 0
            for word in reversed(words):
                if carry_len + len(word) + 1 > overlap:
                    break
                carry.insert(0, word)
                carry_len += len(word) + 1
            current = " ".join(carry)
        current = f"{current}\n{part}".strip()
    if current:
        chunks.append(current)
    return chunks


def make_chunks(parts: list[DocumentPart], filename: str, document_id: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for part in parts:
        heading = part.metadata.get("heading")
        text = part.text.strip()
        if heading and not text.startswith(str(heading)):
            text = f"{heading}\n{text}"
        for piece in split_text(text):
            metadata = {
                "filename": filename,
                "source": filename,
                "document_id": document_id,
                "chunk": len(chunks),
                **part.metadata,
            }
            chunks.append(Chunk(piece, metadata))
    return chunks
