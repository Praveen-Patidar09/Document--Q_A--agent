"""Connectivity probe for the configured services. Never prints credentials or index hosts."""

import json

from app.answerer import OllamaAnswerer
from app.config import settings
from app.embeddings import GeminiEmbeddings
from app.pinecone_store import PineconeDocumentStore


def check() -> dict:
    result = {
        "configured_model": settings.ollama_model,
        "gemini_key_configured": bool(settings.gemini_api_key),
        "pinecone_credentials_configured": bool(settings.pinecone_api_key and settings.pinecone_index_host),
        "gemini": {"status": "not_checked"},
        "pinecone": {"status": "not_checked"},
        "ollama": {"status": "not_checked"},
    }
    try:
        vector = GeminiEmbeddings().query("service connectivity check")
        result["gemini"] = {"status": "ok", "dimension": len(vector)}
    except Exception as exc:
        result["gemini"] = {"status": "unavailable", "error_type": type(exc).__name__}
    try:
        details = PineconeDocumentStore().validate_index()
        result["pinecone"] = {"status": "ok", "dimension": details["dimension"],
                              "metric": details["metric"]}
    except Exception as exc:
        result["pinecone"] = {"status": "unavailable", "error_type": type(exc).__name__}
    try:
        reachable, installed = OllamaAnswerer().check()
        result["ollama"] = {"status": "ok" if reachable and installed else "unavailable",
                            "reachable": reachable, "configured_model_installed": installed}
    except Exception as exc:
        result["ollama"] = {"status": "unavailable", "error_type": type(exc).__name__}
    return result


if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
