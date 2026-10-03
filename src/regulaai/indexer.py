"""
Embed chunks via Ollama (BGE-M3) and store them in Qdrant.
"""

from __future__ import annotations

import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from regulaai.chunker import Chunk
from regulaai.config import get_settings

COLLECTION_NAME = "regulaai_docs"
VECTOR_SIZE = 1024  # BGE-M3 output dimension


def _ollama_url() -> str:
    s = get_settings()
    url = s.ollama_base_url.rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    return url


async def embed_texts(texts: list[str], model: str = "bge-m3") -> list[list[float]]:
    """Embed a batch of texts using Ollama."""
    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.post(
            f"{_ollama_url()}/api/embed",
            json={"model": model, "input": texts},
        )
        resp.raise_for_status()
        data = resp.json()
    return data["embeddings"]


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url="http://localhost:6333")


def recreate_collection(client: QdrantClient) -> None:
    """Drop and recreate the collection (fresh index)."""
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )


async def index_chunks(
    chunks: list[Chunk],
    batch_size: int = 32,
    fresh: bool = True,
) -> int:
    """
    Embed chunks in batches and upsert into Qdrant.

    Returns the number of points indexed.
    """
    client = get_qdrant_client()

    if fresh:
        recreate_collection(client)

    total = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c.text for c in batch]
        vectors = await embed_texts(texts)

        points = [
            PointStruct(
                id=i + j,
                vector=vectors[j],
                payload={
                    "text": c.text,
                    "source": c.source,
                    "page": c.page,
                    "chunk_index": c.chunk_index,
                    "char_len": c.metadata.get("char_len", len(c.text)),
                },
            )
            for j, c in enumerate(batch)
        ]
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        total += len(points)
        print(f"  indexed {total}/{len(chunks)} chunks", end="\r")

    print(f"  indexed {total}/{len(chunks)} chunks  ")
    return total
