"""Gemini embedding adapter, including retry and vector-shape checks."""

import logging
import random
import time
import re
import httpx

from app.config import settings

logger = logging.getLogger(__name__)
RETRYABLE_CODES = {408, 429, 500, 502, 503, 504}


def _retry(call, attempts: int = 6):
    from google.genai import errors

    for attempt in range(attempts):
        try:
            return call()
        except Exception as exc:
            retryable = isinstance(exc, httpx.TransportError)
            if isinstance(exc, errors.APIError):
                retryable = exc.code in RETRYABLE_CODES
            if not retryable or attempt + 1 == attempts:
                raise
            delay = min(8.0, (2 ** attempt) * random.uniform(0.8, 1.2))
            if isinstance(exc, errors.APIError) and exc.code == 429:
                m = re.search(r"retry in ([\d.]+)s", str(exc))
                delay = (float(m.group(1)) if m else 20.0) + 1.0


class GeminiEmbeddings:
    def __init__(self):
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured.")
        from google import genai

        self.client = genai.Client(api_key=settings.gemini_api_key)

    def _embed(self, contents, task_type: str) -> list[list[float]]:
        from google.genai import types

        def call():
            return self.client.models.embed_content(
                model=settings.embedding_model,
                contents=contents,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=settings.embedding_dimension,
                ),
            )

        result = _retry(call)
        vectors = [list(item.values) for item in (result.embeddings or [])]
        expected = len(contents) if isinstance(contents, list) else 1
        if len(vectors) != expected or any(len(vector) != settings.embedding_dimension for vector in vectors):
            raise RuntimeError("Gemini returned an unexpected embedding count or dimension.")
        return vectors

    def documents(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        batch = 25                     # pehle 64 tha
        for offset in range(0, len(texts), batch):
            if offset:
                time.sleep(17)         # 25 chunks / 17s = ~88 per minute, limit 100 ke neeche
            vectors.extend(self._embed(texts[offset:offset + batch], "RETRIEVAL_DOCUMENT"))
        return vectors

    def query(self, text: str) -> list[float]:
        return self._embed(text, "RETRIEVAL_QUERY")[0]
