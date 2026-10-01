"""
Sentence-aware text chunker for RAG.

Splits long text into overlapping chunks, respecting sentence boundaries.
Each chunk carries metadata linking it back to its source document and page.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from regulaai.loader import Document


@dataclass
class Chunk:
    """A retrievable unit of text with provenance."""

    text: str
    source: str
    page: int
    chunk_index: int
    metadata: dict

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return (
            f"Chunk(source={self.source!r}, page={self.page}, "
            f"idx={self.chunk_index}, text={preview!r}...)"
        )


# Sentence boundary: split after ., !, ? when followed by space + capital,
# or on paragraph breaks.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])|\n\s*\n")


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, keeping them intact."""
    text = text.strip()
    if not text:
        return []
    parts = _SENTENCE_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(
    text: str,
    source: str,
    page: int,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    base_metadata: dict | None = None,
) -> list[Chunk]:
    """
    Split text into overlapping chunks, respecting sentence boundaries.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: list[Chunk] = []
    current: list[str] = []
    current_len = 0
    idx = 0

    for sentence in sentences:
        sent_len = len(sentence)

        if current and current_len + sent_len + 1 > chunk_size:
            chunk_str = " ".join(current).strip()
            chunks.append(
                Chunk(
                    text=chunk_str,
                    source=source,
                    page=page,
                    chunk_index=idx,
                    metadata={**(base_metadata or {}), "char_len": len(chunk_str)},
                )
            )
            idx += 1

            # Build overlap from tail of current
            overlap: list[str] = []
            overlap_len = 0
            for s in reversed(current):
                if overlap_len + len(s) > chunk_overlap:
                    break
                overlap.insert(0, s)
                overlap_len += len(s) + 1
            current = overlap
            current_len = overlap_len

        current.append(sentence)
        current_len += sent_len + 1

    if current:
        chunk_str = " ".join(current).strip()
        chunks.append(
            Chunk(
                text=chunk_str,
                source=source,
                page=page,
                chunk_index=idx,
                metadata={**(base_metadata or {}), "char_len": len(chunk_str)},
            )
        )

    return chunks


def chunk_documents(
    docs: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Chunk]:
    """Chunk a list of Documents into a flat list of Chunks."""
    all_chunks: list[Chunk] = []
    for doc in docs:
        all_chunks.extend(
            chunk_text(
                text=doc.text,
                source=doc.source,
                page=doc.page,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                base_metadata=doc.metadata,
            )
        )
    return all_chunks
