"""HTTP boundary for document-grounded question answering."""

from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import logging
from pathlib import PurePosixPath

from fastapi import FastAPI, File, HTTPException, Path, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.answerer import NOT_FOUND, OllamaAnswerer
from app.chunking import make_chunks
from app.config import settings
from app.diagnostics import trace
from app.embeddings import GeminiEmbeddings
from app.parsers import SUPPORTED, extract_document
from app.pinecone_store import PineconeDocumentStore
from app.retrieval import prepare_context, rank_candidates
from app.schemas import QuestionRequest

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)
app = FastAPI(title="Document Q&A Agent", version="3.0.0",
              description="Answers grounded in one selected document.")
app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins),
                   allow_credentials=False, allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
                   allow_headers=["Content-Type"])


@lru_cache(maxsize=1)
def get_store() -> PineconeDocumentStore:
    return PineconeDocumentStore()


@lru_cache(maxsize=1)
def get_embeddings() -> GeminiEmbeddings:
    return GeminiEmbeddings()


@lru_cache(maxsize=1)
def get_answerer() -> OllamaAnswerer:
    return OllamaAnswerer()


def _service_unavailable(service: str, exc: Exception) -> HTTPException:
    logger.exception("%s operation failed", service)
    return HTTPException(status_code=503, detail=f"{service} is unavailable. Check its configuration and try again.")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    checks = {"gemini_configured": bool(settings.gemini_api_key), "pinecone": False,
              "ollama": False, "ollama_model": False}
    try:
        get_store().validate_index()
        checks["pinecone"] = True
    except Exception:
        logger.warning("Pinecone readiness check failed.", exc_info=True)
    try:
        checks["ollama"], checks["ollama_model"] = get_answerer().check()
    except Exception:
        logger.warning("Ollama readiness check failed.", exc_info=True)
    complete = all(checks.values())
    return JSONResponse(status_code=200 if complete else 503,
                        content={"status": "ready" if complete else "not_ready", "checks": checks,
                                 "embedding_model": settings.embedding_model,
                                 "embedding_dimension": settings.embedding_dimension,
                                 "ollama_model": settings.ollama_model})


@app.post("/documents", status_code=201)
def upload_document(response: Response, file: UploadFile = File(...)):
    max_bytes = settings.max_upload_mb * 1024 * 1024
    content = file.file.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise HTTPException(413, f"File is too large. Maximum upload size is {settings.max_upload_mb} MB.")
    if not content:
        raise HTTPException(422, "The uploaded file is empty.")

    filename = PurePosixPath((file.filename or "").replace("\\", "/")).name[:255]
    suffix = PurePosixPath(filename).suffix.lower()
    if suffix not in SUPPORTED:
        raise HTTPException(415, f"Unsupported file type. Supported: {', '.join(sorted(SUPPORTED))}")
    try:
        parts = extract_document(filename, content)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if not parts:
        if suffix == ".pdf":
            raise HTTPException(422, "No selectable text was found. Scanned PDFs require OCR and are not supported.")
        raise HTTPException(422, "No usable text could be extracted from this file.")
    trace("extract", filename=filename, parts=[{"metadata": part.metadata, "text": part.text}
                                                for part in parts])

    document_id = hashlib.sha256(filename.encode("utf-8") + b"\0" + content).hexdigest()[:24]
    chunks = make_chunks(parts, filename, document_id)
    trace("chunk", document_id=document_id,
          chunks=[{"metadata": chunk.metadata, "text": chunk.text} for chunk in chunks])
    if not chunks:
        raise HTTPException(422, "No usable text could be extracted from this file.")
    if len(chunks) > settings.max_chunks:
        raise HTTPException(413, f"This document has {len(chunks)} chunks; the limit is {settings.max_chunks}.")

    try:
        store = get_store()
        if store.document_exists(document_id):
            response.status_code = 200
            return {"document_id": document_id, "filename": filename, "chunks": len(chunks),
                    "already_indexed": True}
        vectors = get_embeddings().documents([chunk.text for chunk in chunks])
        trace("embed_documents", document_id=document_id, text_count=len(chunks),
              vector_count=len(vectors), dimensions=[len(vector) for vector in vectors])
        if len(vectors) != len(chunks):
            raise RuntimeError("Embedding service did not return one vector for every chunk.")
        store.add_document(document_id, filename, chunks, vectors,
                           datetime.now(timezone.utc).isoformat())
        trace("index_verified", document_id=document_id, chunks=len(chunks),
              ready=store.document_exists(document_id))
    except HTTPException:
        raise
    except Exception as exc:
        raise _service_unavailable("Document indexing", exc) from exc
    return {"document_id": document_id, "filename": filename, "chunks": len(chunks)}


@app.get("/documents")
def list_documents():
    try:
        return {"documents": get_store().list_documents()}
    except Exception as exc:
        raise _service_unavailable("Document listing", exc) from exc


@app.delete("/documents/{document_id}", status_code=204)
def delete_document(document_id: str = Path(pattern=r"^[a-f0-9]{24}$")):
    try:
        deleted = get_store().delete_document(document_id)
    except Exception as exc:
        raise _service_unavailable("Document deletion", exc) from exc
    if not deleted:
        raise HTTPException(404, "Document not found or already deleted.")
    return Response(status_code=204)


@app.post("/questions")
def ask_question(body: QuestionRequest):
    trace("question", document_id=body.document_id, question=body.question)
    try:
        store = get_store()
        if not store.document_exists(body.document_id):
            raise HTTPException(404, "Selected document is not indexed or is not ready.")
        vector = get_embeddings().query(body.question)
        trace("embed_query", document_id=body.document_id, dimensions=len(vector))
        rows = store.search(body.document_id, vector, settings.candidate_k)
        trace("pinecone_candidates", document_id=body.document_id,
              candidates=[{"score": score, "metadata": chunk.metadata, "text": chunk.text}
                          for chunk, score in rows])
    except HTTPException:
        raise
    except Exception as exc:
        raise _service_unavailable("Document retrieval", exc) from exc

    candidates = rank_candidates(body.question, rows, settings.context_k, settings.min_similarity)
    trace("ranked_candidates", document_id=body.document_id,
          candidates=[{"similarity": item.similarity, "lexical": item.lexical_score,
                       "rank": item.rank_score, "metadata": item.chunk.metadata,
                       "text": item.chunk.text} for item in candidates])
    if not candidates:
        return {"answer": NOT_FOUND, "sources": []}
    context = prepare_context(candidates, body.question)
    trace("answer_context", document_id=body.document_id, context=context)
    if not context:
        return {"answer": NOT_FOUND, "sources": []}
    try:
        answer = get_answerer().answer(body.question, context)
        trace("answer_return", document_id=body.document_id, answer=answer)
    except Exception as exc:
        raise _service_unavailable("Answer generation", exc) from exc
    sources = []
    for item in candidates:
        metadata = item.chunk.metadata
        source = {key: metadata[key] for key in
                  ("filename", "chunk", "page", "slide", "sheet", "section", "title", "headings", "columns", "structure")
                  if key in metadata}
        source["filename"] = source.get("filename", "Document")
        source["similarity"] = round(item.similarity, 4)
        sources.append(source)
    return {"answer": answer or NOT_FOUND, "sources": sources}
