"""Quick test of the PDF loader."""

from pathlib import Path

from rich.console import Console

from regulaai.loader import load_directory

console = Console()


def main() -> None:
    docs = load_directory("data/raw")
    console.print(f"[bold]Loaded {len(docs)} pages total[/bold]\n")

    # Group by source
    by_source: dict[str, int] = {}
    for d in docs:
        by_source[d.source] = by_source.get(d.source, 0) + 1

    for source, count in sorted(by_source.items()):
        console.print(f"  {source:25s} {count:>4d} pages")

    console.print("\n[bold]First page preview from GDPR:[/bold]")
    for d in docs:
        if d.source == "gdpr.pdf":
            console.print(f"[dim]Page {d.page}:[/dim]")
            console.print(d.text[:500])
            break


if __name__ == "__main__":
    main()
