"""Pinecone persistence. One namespace per document and full-vector-set readiness checks."""

import time
from functools import cached_property
from typing import Any

from app.config import settings
from app.models import Chunk


class PineconeDocumentStore:
    def __init__(self, client=None, index=None):
        self._client = client
        self._index = index

    @cached_property
    def client(self):
        if self._client is not None:
            return self._client
        if not settings.pinecone_api_key or not settings.pinecone_index_host:
            raise RuntimeError("PINECONE_API_KEY and PINECONE_INDEX_HOST must be configured.")
        from pinecone import Pinecone

        return Pinecone(api_key=settings.pinecone_api_key)

    @cached_property
    def index(self):
        if self._index is not None:
            return self._index
        return self.client.Index(host=settings.pinecone_index_host)

    @staticmethod
    def _field(value: Any, key: str, default=None):
        if isinstance(value, dict):
            return value.get(key, default)
        return getattr(value, key, default)

    def validate_index(self) -> dict:
        stats = self.index.describe_index_stats()
        dimension = self._field(stats, "dimension")
        if dimension is not None and int(dimension) != settings.embedding_dimension:
            raise RuntimeError(f"Pinecone dimension is {dimension}, expected {settings.embedding_dimension}.")
        if settings.pinecone_index_name:
            description = self.client.describe_index(settings.pinecone_index_name)
            metric = self._field(description, "metric")
            desc_dimension = self._field(description, "dimension")
            if metric and str(metric).lower() != settings.pinecone_metric:
                raise RuntimeError(f"Pinecone metric is {metric}, expected {settings.pinecone_metric}.")
            if desc_dimension and int(desc_dimension) != settings.embedding_dimension:
                raise RuntimeError(f"Pinecone dimension is {desc_dimension}, expected {settings.embedding_dimension}.")
        return {"dimension": dimension or settings.embedding_dimension,
                "metric": settings.pinecone_metric,
                "namespaces": self._field(stats, "namespaces", {}) or {}}

    def _stats(self) -> dict:
        stats = self.index.describe_index_stats()
        return self._field(stats, "namespaces", {}) or {}

    def document_exists(self, document_id: str) -> bool:
        summary = self._stats().get(document_id)
        if not summary or int(self._field(summary, "vector_count", 0) or 0) < 1:
            return False
        manifest_id = f"{document_id}-manifest"
        fetched = self.index.fetch(ids=[manifest_id], namespace=document_id)
        manifest = (self._field(fetched, "vectors", {}) or {}).get(manifest_id)
        if not manifest:
            # Backward compatibility for complete namespaces written by the previous backend.
            return True
        metadata = self._field(manifest, "metadata", {}) or {}
        expected = int(metadata.get("chunk_count", 0)) + 1
        actual = int(self._field(summary, "vector_count", 0) or 0)
        return metadata.get("record_type") == "manifest" and expected > 1 and actual >= expected

    def add_document(self, document_id: str, filename: str, chunks: list[Chunk],
                     vectors: list[list[float]], created_at: str) -> None:
        if not chunks or len(chunks) != len(vectors):
            raise ValueError("Each document chunk must have exactly one embedding.")
        if any(len(vector) != settings.embedding_dimension for vector in vectors):
            raise ValueError("Embedding dimension does not match the Pinecone index.")
        records = []
        for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
            metadata = {key: value for key, value in chunk.metadata.items()
                        if value is not None and value != [] and isinstance(value, (str, int, float, bool, list))}
            metadata.update({"record_type": "chunk", "text": chunk.text, "created_at": created_at,
                             "chunk_count": len(chunks)})
            records.append({"id": f"{document_id}-c{index:06d}", "values": vector, "metadata": metadata})
        manifest = {"id": f"{document_id}-manifest", "values": vectors[0], "metadata": {
            "record_type": "manifest", "document_id": document_id, "filename": filename,
            "created_at": created_at, "chunk_count": len(chunks),
        }}
        all_records = records + [manifest]
        try:
            for offset in range(0, len(all_records), 100):
                self.index.upsert(vectors=all_records[offset:offset + 100], namespace=document_id)
            expected = {item["id"] for item in all_records}
            # Verify every uploaded chunk and the manifest, not just one representative vector.
            for delay in (0, 0.5, 1, 2, 3, 4):
                if delay:
                    time.sleep(delay)
                actual = set()
                for offset in range(0, len(expected), 100):
                    ids = list(expected)[offset:offset + 100]
                    fetched = self.index.fetch(ids=ids, namespace=document_id)
                    actual.update((self._field(fetched, "vectors", {}) or {}).keys())
                if actual == expected:
                    return
            raise TimeoutError("Pinecone did not make every indexed chunk readable.")
        except Exception:
            try:
                self.index.delete(delete_all=True, namespace=document_id)
            except Exception:
                pass
            raise

    def search(self, document_id: str, vector: list[float], top_k: int):
        result = self.index.query(
            vector=vector, top_k=top_k, namespace=document_id,
            include_metadata=True,
        )
        candidates = []
        for item in self._field(result, "matches", []) or []:
            metadata = self._field(item, "metadata", {}) or {}
            if metadata.get("document_id") not in (None, document_id):
                continue
            if metadata.get("record_type") not in (None, "chunk") or not metadata.get("text"):
                continue
            score = float(self._field(item, "score", 0.0) or 0.0)
            candidates.append((Chunk(metadata["text"], metadata), score))
        return candidates

    def list_documents(self) -> list[dict]:
        docs = []
        for document_id, summary in self._stats().items():
            if document_id == "__default__":
                continue
            first = next(iter(self.index.list(namespace=document_id)), None)
            ids = self._field(first, "vectors", []) or []
            manifest_id = f"{document_id}-manifest"
            if manifest_id not in ids:
                # list() can paginate; fetch a deterministic known manifest directly.
                fetched = self.index.fetch(ids=[manifest_id], namespace=document_id)
                vectors = self._field(fetched, "vectors", {}) or {}
                record = vectors.get(manifest_id)
            else:
                fetched = self.index.fetch(ids=[manifest_id], namespace=document_id)
                record = (self._field(fetched, "vectors", {}) or {}).get(manifest_id)
            if not record:
                # Preserve visibility and query support for namespaces written by the previous
                # backend (which did not include a manifest record).
                first_page = next(iter(self.index.list(namespace=document_id)), None)
                old_ids = self._field(first_page, "vectors", []) or []
                if not old_ids:
                    continue
                first_id = self._field(old_ids[0], "id", old_ids[0])
                old_record = self.index.fetch(ids=[first_id], namespace=document_id)
                record = (self._field(old_record, "vectors", {}) or {}).get(first_id)
                if not record:
                    continue
                metadata = self._field(record, "metadata", {}) or {}
                docs.append({"document_id": document_id,
                             "filename": metadata.get("filename", metadata.get("source", "Document")),
                             "chunks": metadata.get("chunk_count", self._field(summary, "vector_count", 0)),
                             "created_at": metadata.get("created_at")})
                continue
            metadata = self._field(record, "metadata", {}) or {}
            count = int(metadata.get("chunk_count", 0))
            actual = int(self._field(summary, "vector_count", 0) or 0)
            if metadata.get("record_type") != "manifest" or actual < count + 1:
                continue
            docs.append({"document_id": document_id, "filename": metadata.get("filename", "Document"),
                         "chunks": count, "created_at": metadata.get("created_at")})
        return sorted(docs, key=lambda item: item.get("created_at") or "", reverse=True)

    def delete_document(self, document_id: str) -> bool:
        if not self.document_exists(document_id):
            return False
        self.index.delete(delete_all=True, namespace=document_id)
        for delay in (0.25, 0.5, 1, 2, 3, 4):
            time.sleep(delay)
            if document_id not in self._stats():
                return True
        raise TimeoutError("Pinecone accepted deletion but the document namespace remains.")
