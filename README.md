# Document Q&A Agent

Ask questions about one uploaded document. The backend extracts text, embeds and indexes chunks in Pinecone, retrieves only from the selected document namespace, and asks the configured local Ollama model for an answer with exact supporting quotes. Every quote must occur in retrieved text; numeric claims must also occur in the quoted evidence. Answers are concise professional English. Unsupported or malformed answers fail closed with:

`This information is not available in the selected document.`

## Services and defaults

- Embeddings: Gemini `gemini-embedding-001`, 768 dimensions.
- Vector store: Pinecone dense index, cosine metric, one namespace per document.
- Answer generator: local Ollama `qwen3:4b-instruct-2507-q4_K_M` (settings can select another installed model).
- Backend API: FastAPI.
- Frontend: existing React/Vite app; backend contract remains `/health`, `/documents`, and `/questions`.

The frontend source, assets, scripts, and configuration are independent of the backend and were left unchanged.

## Supported files

PDF (selectable text only), TXT, Markdown, DOCX, PPTX, XLSX, CSV, and HTML. Pages, slides, sheets, paragraph/heading boundaries, and source locations are retained where available. Scanned PDFs need OCR and are rejected. Upload limit defaults to 8 MB; office archives are limited to 100 MB unpacked.

## Configuration

Copy `.env.example` to `.env` once and set `GEMINI_API_KEY`, `PINECONE_API_KEY`, and `PINECONE_INDEX_HOST`. The application loads `.env` at runtime; it is ignored by Git. Configure the Pinecone index with 768 dimensions and cosine metric to match Gemini embeddings. Optional `PINECONE_INDEX_NAME` enables control-plane metric/dimension validation.

Relevant settings include `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OLLAMA_NUM_CTX`, `OLLAMA_TIMEOUT`, `LLM_TEMPERATURE`, `LLM_MAX_OUTPUT_TOKENS`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `CANDIDATE_K`, `CONTEXT_K`, and `MIN_SIMILARITY`.

## Run on Windows

From the project root, in PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

In another terminal:

```powershell
cd frontend
npm run dev
```

The frontend forwards API calls to `http://127.0.0.1:8000`. API docs are at `http://127.0.0.1:8000/docs`; `/ready` reports whether Gemini configuration, Pinecone, Ollama, and the configured model are ready.

To install the declared dependencies into a fresh virtual environment:

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Verification

```powershell
$env:TEMP = (Get-Location).Path
$env:TMP = (Get-Location).Path
python -m unittest discover -s tests -v
python -m eval.run_eval
python -m eval.run_eval --live
python -m eval.check_services
```

The deterministic evaluation uses known passages and fixed model replies; it does not claim live model accuracy. `--live` invokes the configured Ollama model. Gemini and Pinecone calls require their live credentials and network access.

## API

- `GET /health`: process health.
- `GET /ready`: service readiness and configured model details.
- `POST /documents`: multipart upload with form field `file`; returns document ID and chunk count.
- `GET /documents`: indexed documents.
- `DELETE /documents/{document_id}`: remove the selected document namespace.
- `POST /questions`: JSON `{ "question": "...", "document_id": "24-hex-id" }`; returns `{ "answer": "...", "sources": [...] }`.

Each question is independent. There is no cross-document search or chat memory.
