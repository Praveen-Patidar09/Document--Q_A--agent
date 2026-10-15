"""Validated runtime configuration. Values are loaded from environment and the local .env file."""

from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()


def _int(name: str, default: int, minimum: int = 1) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer.") from exc
    if value < minimum:
        raise RuntimeError(f"{name} must be at least {minimum}.")
    return value


def _float(name: str, default: float, minimum: float | None = None) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number.") from exc
    if minimum is not None and value < minimum:
        raise RuntimeError(f"{name} must be at least {minimum}.")
    return value


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
    embedding_dimension: int = _int("PINECONE_DIMENSION", 768)
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_index_host: str = os.getenv("PINECONE_INDEX_HOST", "")
    pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME", "")
    pinecone_metric: str = os.getenv("PINECONE_METRIC", "cosine").lower()
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen3:4b-instruct-2507-q4_K_M")
    ollama_num_ctx: int = _int("OLLAMA_NUM_CTX", 8192)
    ollama_num_predict: int = _int("LLM_MAX_OUTPUT_TOKENS", 512)
    ollama_timeout: float = _float("OLLAMA_TIMEOUT", 120, 1)
    ollama_keep_alive: str = os.getenv("OLLAMA_KEEP_ALIVE", "15m")
    temperature: float = _float("LLM_TEMPERATURE", 0.0, 0)
    max_upload_mb: int = _int("MAX_UPLOAD_MB", 8)
    max_chunks: int = _int("MAX_CHUNKS", 2500)
    chunk_size: int = _int("CHUNK_SIZE", 1100)
    chunk_overlap: int = _int("CHUNK_OVERLAP", 120, 0)
    candidate_k: int = _int("CANDIDATE_K", 16)
    context_k: int = _int("CONTEXT_K", 5)
    min_similarity: float = _float("MIN_SIMILARITY", 0.42, -1)
    diagnostics: bool = os.getenv("DOCQA_DIAGNOSTICS", "false").strip().lower() in ("1", "true", "yes", "on")
    cors_origins: tuple[str, ...] = tuple(
        item.strip() for item in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
        if item.strip()
    )

    def __post_init__(self):
        if self.embedding_dimension not in (768, 1536, 3072):
            raise RuntimeError("PINECONE_DIMENSION must be 768, 1536, or 3072 for gemini-embedding-001.")
        if self.pinecone_metric != "cosine":
            raise RuntimeError("This retrieval configuration requires a cosine Pinecone index.")
        if self.min_similarity > 1:
            raise RuntimeError("MIN_SIMILARITY must not exceed 1 for a cosine index.")
        if self.chunk_overlap >= self.chunk_size:
            raise RuntimeError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")


settings = Settings()
