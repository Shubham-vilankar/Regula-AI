"""
RAG-grounded chat.

How it Works , 
Pipeline:
  1. Embed the user query
  2. Retrieve top-k relevant chunks from Qdrant
  3. Build a prompt containing those chunks as context
  4. Ask the LLM (with thinking) to answer using ONLY the context
  5. Return the answer plus the sources it was grounded in (Ground truth answers from source PDF's/Docs)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from regulaai.llm_client import OllamaClient
from regulaai.retriever import RetrievedChunk, retrieve
from regulaai.thinking import ThinkingResponse


SYSTEM_PROMPT = """You are RegulaAI, a compliance assistant.

You answer questions using ONLY the numbered source excerpts provided in the
user message. Follow these rules strictly:

1. If the sources do not contain the answer, say so explicitly.
2. Cite sources inline using the format [1], [2], etc., matching the source numbers.
3. Never invent facts that are not in the sources.
4. If sources contradict each other, point that out.
5. Be concise and direct.
"""


@dataclass
class RAGResponse:
    answer: str
    thinking: str
    sources: list[RetrievedChunk]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    trace_id: str | None = None
    raw: ThinkingResponse | None = field(default=None, repr=False)


def _format_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks as numbered source excerpts."""
    lines = []
    for i, c in enumerate(chunks, 1):
        lines.append(
            f"[{i}] (source: {c.source}, page {c.page}, "
            f"relevance {c.score:.2f})\n{c.text}\n"
        )
    return "\n".join(lines)


async def rag_answer(
    query: str,
    top_k: int = 5,
    client: OllamaClient | None = None,
    trace_name: str = "rag-chat",
) -> RAGResponse:
    """Answer a query grounded in retrieved regulatory documents."""
    # 1. Retrieve
    chunks = await retrieve(query, top_k=top_k)

    # 2. Build the grounded prompt
    context = _format_context(chunks)
    user_prompt = (
        f"SOURCES:\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        f"Answer using only the sources above and cite them as [1], [2], etc."
    )

    # 3. Ask the LLM
    client = client or OllamaClient()
    resp = await client.chat(
        prompt=user_prompt,
        system=SYSTEM_PROMPT,
        enable_thinking=True,
        trace_name=trace_name,
    )

    return RAGResponse(
        answer=resp.answer,
        thinking=resp.thinking,
        sources=chunks,
        prompt_tokens=resp.prompt_tokens,
        completion_tokens=resp.completion_tokens,
        trace_id=resp.trace_id,
        raw=resp,
    )
