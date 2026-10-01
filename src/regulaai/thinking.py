"""
Chain-of-Thought parser for Ollama responses.

Ollama returns thinking and answer as separate JSON fields, so we
just need a clean data structure to carry both.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ChunkKind(str, Enum):
    THINKING = "thinking"
    ANSWER = "answer"


@dataclass(frozen=True)
class ThinkingChunk:
    kind: ChunkKind
    text: str


@dataclass
class ThinkingResponse:
    thinking: str
    answer: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    trace_id: str | None = None
