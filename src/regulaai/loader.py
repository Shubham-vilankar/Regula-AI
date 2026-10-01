"""
PDF loader for regulatory documents.

Reads PDFs, extracts text per page, and returns structured Document objects
with metadata (source, page number) for downstream chunking and citation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader


@dataclass
class Document:
    """A single page of text from a source PDF."""

    text: str
    source: str          # filename
    page: int            # 1-indexed
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return f"Document(source={self.source!r}, page={self.page}, text={preview!r}...)"


def load_pdf(path: Path | str) -> list[Document]:
    """Load a PDF and return a list of Documents, one per non-empty page."""
    path = Path(path)
    reader = PdfReader(str(path))
    docs: list[Document] = []

    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if not text:
            continue
        docs.append(
            Document(
                text=text,
                source=path.name,
                page=i,
                metadata={"total_pages": len(reader.pages)},
            )
        )

    return docs


def load_directory(directory: Path | str) -> list[Document]:
    """Load every PDF in a directory."""
    directory = Path(directory)
    all_docs: list[Document] = []
    for pdf_path in sorted(directory.glob("*.pdf")):
        all_docs.extend(load_pdf(pdf_path))
    return all_docs
