"""
Smoke test: verify the Ollama client returns thinking + answer.

Run:
    uv run python scripts/smoke_test.py
"""

import asyncio
import sys

from rich.console import Console
from rich.panel import Panel

from regulaai.llm_client import OllamaClient

console = Console()

PROMPT = (
    "A user asks: 'Can we share customer data with a partner in Germany?'\n"
    "Think step by step, then give a short answer."
)


async def main() -> None:
    try:
        client = OllamaClient()
        console.rule("[bold cyan]Calling Ollama")
        resp = await client.chat(prompt=PROMPT, enable_thinking=True)

        console.print(
            Panel(
                resp.thinking or "(no thinking captured)",
                title="Thinking",
                border_style="yellow",
            )
        )
        console.print(
            Panel(
                resp.answer or "(no answer)",
                title="Answer",
                border_style="green",
            )
        )
        console.print(
            f"[dim]prompt_tokens={resp.prompt_tokens} "
            f"completion_tokens={resp.completion_tokens}[/dim]"
        )
    except Exception as e:
        console.print(f"[bold red]FAILED:[/bold red] {e}")
        console.print("\n[dim]Checklist:[/dim]")
        console.print("  - Is Ollama running? (ollama list)")
        console.print("  - Is qwen3:14b pulled?")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
