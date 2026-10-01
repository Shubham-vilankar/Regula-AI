"""Quick test of the chunker."""

from rich.console import Console

from regulaai.chunker import chunk_documents
from regulaai.loader import load_directory

console = Console()


def main() -> None:
    docs = load_directory("data/raw")
    console.print(f"[bold]Loaded {len(docs)} pages[/bold]")

    chunks = chunk_documents(docs, chunk_size=1000, chunk_overlap=200)
    console.print(f"[bold]Produced {len(chunks)} chunks[/bold]\n")

    lens = [len(c.text) for c in chunks]
    console.print(f"  avg chunk size: {sum(lens) // len(lens)} chars")
    console.print(f"  min chunk size: {min(lens)} chars")
    console.print(f"  max chunk size: {max(lens)} chars\n")

    console.print("[bold]Sample chunks from GDPR:[/bold]")
    gdpr_chunks = [c for c in chunks if c.source == "gdpr.pdf"][:2]
    for c in gdpr_chunks:
        console.print(f"[dim]--- page {c.page}, chunk {c.chunk_index} ---[/dim]")
        console.print(c.text[:400])
        console.print()


if __name__ == "__main__":
    main()
