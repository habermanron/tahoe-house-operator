import hashlib
import os
import re
from typing import Any

from openai import AsyncOpenAI

from tahoe_operator.config import get_settings

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")


class CorpusService:
    """Local Chroma vector index; embeddings are requested only through the configured gateway."""

    def __init__(self) -> None:
        settings = get_settings()
        self.corpus_path = settings.corpus_path
        self.store_path = settings.vector_store_path
        self.embedding_model = settings.embedding_model
        self.base_url = settings.model_gateway_base_url
        self.api_key = settings.model_gateway_api_key

    @staticmethod
    def _chunks(text: str, size: int = 900, overlap: int = 120) -> list[str]:
        words = text.split()
        chunks: list[str] = []
        cursor = 0
        while cursor < len(words):
            part = words[cursor : cursor + size]
            if not part:
                break
            chunks.append(" ".join(part))
            cursor += max(1, size - overlap)
        return chunks

    def _documents(self) -> list[dict[str, Any]]:
        docs = []
        for path in sorted(self.corpus_path.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            chunks = self._chunks(text)
            for index, chunk in enumerate(chunks):
                docs.append(
                    {
                        "id": f"{path.stem}-{index}",
                        "text": chunk,
                        "metadata": {
                            "source_id": path.stem,
                            "title": path.stem.replace("_", " ").title(),
                            "section": str(index + 1),
                            "version": "fixture-v1",
                            "effective_date": "2026-01-01",
                        },
                    }
                )
        return docs

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        if not (self.base_url and self.api_key and self.embedding_model):
            raise RuntimeError("Embedding gateway is not configured")
        client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
        response = await client.embeddings.create(model=self.embedding_model, input=texts)
        return [item.embedding for item in response.data]

    async def index_documents(self) -> dict[str, Any]:
        import chromadb

        docs = self._documents()
        if not docs:
            return {"status": "empty", "indexed": 0}
        embeddings = await self._embed([doc["text"] for doc in docs])
        self.store_path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(self.store_path))
        collection = client.get_or_create_collection("house_corpus")
        # Replace fixture entries by stable ID so re-indexing is repeatable.
        existing = collection.get()
        if existing.get("ids"):
            collection.delete(ids=existing["ids"])
        collection.add(
            ids=[doc["id"] for doc in docs],
            documents=[doc["text"] for doc in docs],
            metadatas=[doc["metadata"] for doc in docs],
            embeddings=embeddings,
        )
        return {"status": "indexed", "indexed": len(docs), "collection": "house_corpus"}

    async def search(self, query: str, limit: int = 4) -> dict[str, Any]:
        docs = self._documents()
        if not docs:
            return {"status": "empty", "mode": "none", "chunks": []}
        settings = get_settings()
        if self.base_url and self.api_key and self.embedding_model:
            try:
                import chromadb

                self.store_path.mkdir(parents=True, exist_ok=True)
                client = chromadb.PersistentClient(path=str(self.store_path))
                collection = client.get_or_create_collection("house_corpus")
                if collection.count() == 0:
                    await self.index_documents()
                    collection = chromadb.PersistentClient(
                        path=str(self.store_path)
                    ).get_collection("house_corpus")
                embedding = (await self._embed([query]))[0]
                found = collection.query(
                    query_embeddings=[embedding],
                    n_results=limit,
                    include=["documents", "metadatas", "distances"],
                )
                chunks = []
                for i, text in enumerate(found["documents"][0]):
                    metadata = found["metadatas"][0][i]
                    chunks.append(
                        {
                            "text": text,
                            "source_id": metadata["source_id"],
                            "title": metadata["title"],
                            "section": metadata["section"],
                            "version": metadata["version"],
                            "effective_date": metadata["effective_date"],
                            "distance": found["distances"][0][i],
                        }
                    )
                return {"status": "ok", "mode": "vector", "chunks": chunks}
            except Exception as exc:
                if not settings.allow_keyword_rag_fallback:
                    return {
                        "status": "unavailable",
                        "mode": "vector",
                        "reason": str(exc),
                        "chunks": [],
                    }
        terms = {term.lower() for term in re.findall(r"[A-Za-z0-9_]+", query) if len(term) > 2}
        ranked = []
        for doc in docs:
            body = doc["text"].lower()
            score = sum(body.count(term) for term in terms)
            if score:
                ranked.append((score, doc))
        ranked.sort(key=lambda item: item[0], reverse=True)
        chunks = [
            {"text": doc["text"], **doc["metadata"], "score": score}
            for score, doc in ranked[:limit]
        ]
        return {
            "status": "ok" if chunks else "no_match",
            "mode": "keyword_fallback",
            "chunks": chunks,
        }


def fixture_embedding(text: str, dimensions: int = 16) -> list[float]:
    """Stable test-only embedding; production embeddings must come from the model gateway."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [digest[i] / 255.0 for i in range(dimensions)]
