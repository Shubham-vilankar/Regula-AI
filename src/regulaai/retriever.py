"""
Vector retriever: query -> relevant chunks from Qdrant.
"""

from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import QdrantClient

from regulaai.indexer import (
    COLLECTION_NAME,
    embed_texts,
    get_qdrant_client,
)


@dataclass
class RetrievedChunk:
    text: str
    source: str
    page: int
    chunk_index: int
    score: float

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return (
            f"RetrievedChunk(score={self.score:.3f}, source={self.source!r}, "
            f"page={self.page}, text={preview!r}...)"
        )


async def retrieve(
    query: str,
    top_k: int = 5,
    client: QdrantClient | None = None,
) -> list[RetrievedChunk]:
    """Embed the query and return the top_k most similar chunks."""
    client = client or get_qdrant_client()
    [query_vec] = await embed_texts([query])

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        limit=top_k,
        with_payload=True,
    ).points

    out: list[RetrievedChunk] = []
    for r in results:
        payload = r.payload or {}
        out.append(
            RetrievedChunk(
                text=payload.get("text", ""),
                source=payload.get("source", "unknown"),
                page=payload.get("page", 0),
                chunk_index=payload.get("chunk_index", 0),
                score=r.score,
            )
        )
    return out
