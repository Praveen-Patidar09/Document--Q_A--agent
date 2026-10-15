import React, { useCallback, useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");
const ACCEPT = ".pdf,.txt,.md,.docx,.pptx,.xlsx,.csv,.html,.htm";

async function api(path, options = {}) {
  let response;
  try {
    response = await fetch(`${API_BASE}${path}`, options);
  } catch {
    throw new Error("Couldn't reach the backend. Check that it is running and try again.");
  }
  if (!response.ok) {
    let message = `Request failed (${response.status}).`;
    try {
      const body = await response.json();
      message = body.detail || body.message || message;
    } catch { /* Keep the useful HTTP status fallback. */ }
    throw new Error(message);
  }
  if (response.status === 204) return null;
  return response.json();
}

function formatSource(source) {
  const parts = [source.filename || source.source || "Document"];
  if (source.page) parts.push(`page ${source.page}`);
  if (source.slide) parts.push(`slide ${source.slide}`);
  if (source.sheet) parts.push(`sheet ${source.sheet}`);
  if (source.section) parts.push(source.section);
  if (source.title) parts.push(source.title);
  if (source.headings?.length) parts.push(source.headings.slice(0, 2).join(" / "));
  if (source.columns) parts.push(`columns: ${source.columns}`);
  if (source.structure) parts.push(source.structure);
  if (Number.isInteger(source.chunk)) parts.push(`chunk ${source.chunk + 1}`);
  return parts.join(" · ");
}

function App() {
  const [documents, setDocuments] = useState([]);
  const [selectedId, setSelectedId] = useState(() => localStorage.getItem("docwise-selected") || "");
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState([]);
  const [error, setError] = useState("");
  const [loadingList, setLoadingList] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [asking, setAsking] = useState(false);
  const [deletingId, setDeletingId] = useState("");
  const [apiStatus, setApiStatus] = useState("checking");

  const selected = useMemo(() => documents.find((doc) => doc.document_id === selectedId),
    [documents, selectedId]);

  const loadDocuments = useCallback(async () => {
    setLoadingList(true);
    try {
      const result = await api("/documents");
      setDocuments(result.documents || []);
      setError("");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoadingList(false);
    }
  }, []);

  useEffect(() => {
    loadDocuments();
    api("/health").then(() => setApiStatus("online"))
      .catch(() => setApiStatus("offline"));
  }, [loadDocuments]);

  useEffect(() => {
    if (selectedId) localStorage.setItem("docwise-selected", selectedId);
    else localStorage.removeItem("docwise-selected");
  }, [selectedId]);

  useEffect(() => {
    if (selectedId && documents.length && !documents.some((doc) => doc.document_id === selectedId)) {
      setSelectedId("");
    }
  }, [documents, selectedId]);

  function clearConversation() {
    setQuestion("");
    setAnswer("");
    setSources([]);
    setError("");
  }

  async function uploadFile(event) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    setUploading(true);
    setError("");
    setAnswer("");
    setSources([]);
    try {
      const data = new FormData();
      data.append("file", file);
      const result = await api("/documents", { method: "POST", body: data });
      const refreshed = await api("/documents");
      setDocuments(refreshed.documents || []);
      setSelectedId(result.document_id);
      setQuestion("");
    } catch (err) {
      setError(err.message);
    } finally {
      setUploading(false);
    }
  }

  async function submitQuestion(event) {
    event.preventDefault();
    if (!selected || !question.trim() || asking) return;
    setAsking(true);
    setError("");
    setAnswer("");
    setSources([]);
    try {
      const result = await api("/questions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: question.trim(), document_id: selected.document_id }),
      });
      setAnswer(result.answer || "");
      setSources(result.sources || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setAsking(false);
    }
  }

  async function deleteDocument(doc) {
    if (!window.confirm(`Delete “${doc.filename}” and its indexed content?`)) return;
    setDeletingId(doc.document_id);
    setError("");
    try {
      await api(`/documents/${encodeURIComponent(doc.document_id)}`, { method: "DELETE" });
      setDocuments((items) => items.filter((item) => item.document_id !== doc.document_id));
      if (selectedId === doc.document_id) {
        setSelectedId("");
        clearConversation();
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setDeletingId("");
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="Docwise home">
          <span className="brand-mark" aria-hidden="true"><FileIcon /></span>
          <span>docwise<span className="brand-dot">.</span></span>
        </a>
        <div className={`api-indicator ${apiStatus}`} role="status">
          <span className="status-dot" />
          {apiStatus === "checking" ? "Connecting" : apiStatus === "online" ? "System online" : "Backend offline"}
        </div>
      </header>

      <main id="top" className="layout">
        <section className="intro">
          <div className="eyebrow"><span className="eyebrow-line" /> DOCUMENT INTELLIGENCE</div>
          <h1>Your documents,<br /><span>made understandable.</span></h1>
          <p>Upload a file, ask a question, and get an answer grounded in its contents.</p>
        </section>

        <div className="workspace">
          <aside className="library panel">
            <div className="panel-heading">
              <div>
                <span className="step-label">STEP 01</span>
                <h2>Your library</h2>
              </div>
              <span className="count-pill">{documents.length}</span>
            </div>

            <label className={`upload-zone ${uploading ? "is-busy" : ""}`}>
              <input type="file" accept={ACCEPT} onChange={uploadFile} disabled={uploading || asking} />
              <span className="upload-icon"><UploadIcon /></span>
              <span className="upload-copy">
                <strong>{uploading ? "Indexing your document…" : "Add a document"}</strong>
                <small>{uploading ? "Extracting text and preparing search" : "PDF, DOCX, PPTX, XLSX, CSV, HTML, TXT, MD · up to 8 MB"}</small>
              </span>
              {uploading ? <span className="spinner" aria-label="Loading" /> : <span className="upload-arrow">↗</span>}
            </label>

            <div className="list-heading">
              <span>UPLOADED FILES</span>
              <button className="icon-button refresh-button" onClick={loadDocuments} disabled={loadingList} aria-label="Refresh document list"><RefreshIcon /></button>
            </div>

            <div className="document-list" aria-live="polite">
              {loadingList && <div className="list-message"><span className="spinner small" /> Loading library</div>}
              {!loadingList && !documents.length && (
                <div className="empty-library"><span className="empty-file"><FileIcon /></span><strong>Your library is empty</strong><span>Add a file to get started.</span></div>
              )}
              {!loadingList && documents.map((doc) => (
                <div key={doc.document_id} className={`document-row ${selectedId === doc.document_id ? "selected" : ""}`}>
                  <button className="document-select" onClick={() => { setSelectedId(doc.document_id); clearConversation(); }} aria-pressed={selectedId === doc.document_id} disabled={asking || uploading}>
                    <span className="file-type"><FileIcon /></span>
                    <span className="document-meta"><strong title={doc.filename}>{doc.filename}</strong><small>{doc.chunks} text chunks</small></span>
                    {selectedId === doc.document_id && <span className="selected-check">✓</span>}
                  </button>
                  <button className="icon-button delete-button" onClick={() => deleteDocument(doc)} disabled={deletingId === doc.document_id || asking || uploading} aria-label={`Delete ${doc.filename}`} title="Delete document">
                    {deletingId === doc.document_id ? <span className="spinner tiny" /> : <TrashIcon />}
                  </button>
                </div>
              ))}
            </div>
            <div className="privacy-note"><LockIcon /><span>Answers use only the selected document.</span></div>
          </aside>

          <section className="qa panel">
            <div className="panel-heading qa-heading">
              <div><span className="step-label">STEP 02</span><h2>Ask your document</h2></div>
              <span className="sparkle-mark"><SparkleIcon /></span>
            </div>

            <div className={`selected-card ${selected ? "has-selection" : "no-selection"}`}>
              <span className="selected-card-icon"><FileIcon /></span>
              <span className="selected-card-copy">
                <small>SELECTED DOCUMENT</small>
                <strong>{selected ? selected.filename : "Choose a document to begin"}</strong>
                {selected && <span>{selected.chunks} indexed text chunks</span>}
              </span>
              {selected && <span className="ready-label"><span /> READY</span>}
            </div>

            <form className="question-form" onSubmit={submitQuestion}>
              <label htmlFor="question">YOUR QUESTION</label>
              <textarea id="question" value={question} onChange={(event) => setQuestion(event.target.value)} maxLength={2000} placeholder={selected ? "What would you like to know?" : "Select a document first…"} disabled={!selected || asking} rows="4" />
              <div className="form-footer">
                <span>{question.length}/2000</span>
                <div className="form-actions">
                  {(question || answer || error) && <button type="button" className="reset-button" onClick={clearConversation} disabled={asking}>Clear</button>}
                  <button type="submit" className="ask-button" disabled={!selected || !question.trim() || asking}>
                    {asking ? <><span className="spinner button-spinner" /> Thinking…</> : <>Ask question <ArrowIcon /></>}
                  </button>
                </div>
              </div>
            </form>

            {error && <div className="notice error-notice" role="alert"><span className="notice-symbol">!</span><span>{error}</span><button onClick={() => setError("")} aria-label="Dismiss error">×</button></div>}

            {!answer && !asking && !error && (
              <div className="answer-placeholder">
                <span className="answer-placeholder-icon"><SparkleIcon /></span>
                <strong>Your answer will appear here</strong>
                <span>Choose a document and ask a question to get started.</span>
              </div>
            )}

            {asking && <div className="answer-loading"><span className="spinner" /><span><strong>Searching your document</strong><small>Finding the most relevant passages…</small></span></div>}

            {answer && !asking && (
              <article className="answer-card" aria-live="polite">
                <div className="answer-title"><span className="answer-badge"><SparkleIcon /></span><div><span>DOCUMENT ANSWER</span><strong>Here’s what I found</strong></div></div>
                <p className="answer-text">{answer}</p>
                {sources.length > 0 && <div className="sources-block"><div className="sources-heading"><span>GROUNDED IN</span><span>{sources.length} {sources.length === 1 ? "SOURCE" : "SOURCES"}</span></div><ul>{sources.map((source, index) => <li key={`${source.chunk ?? index}-${index}`}><span className="source-number">{index + 1}</span><span>{formatSource(source)}</span></li>)}</ul></div>}
                {sources.length === 0 && <div className="no-source-note">No matching passage was found in this document.</div>}
              </article>
            )}
            <div className="grounding-note"><LockIcon /><span>Responses are generated from the selected document only.</span></div>
          </section>
        </div>

        <footer className="footer"><span>DOCWISE</span><span>Private by design · Document-grounded answers</span></footer>
      </main>
    </div>
  );
}

function FileIcon() { return <svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M6.75 3.75h7l4.5 4.5v12h-11.5z" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round"/><path d="M13.5 3.75v5h4.75M9.5 13h6M9.5 16.5h6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/></svg>; }
function UploadIcon() { return <svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M12 15.5V4m0 0L7.75 8.25M12 4l4.25 4.25M5 15.5v3.25c0 .69.56 1.25 1.25 1.25h11.5c.69 0 1.25-.56 1.25-1.25V15.5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"/></svg>; }
function ArrowIcon() { return <svg viewBox="0 0 20 20" fill="none" aria-hidden="true"><path d="M4 10h11m-4.5-4.5L15 10l-4.5 4.5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"/></svg>; }
function RefreshIcon() { return <svg viewBox="0 0 20 20" fill="none" aria-hidden="true"><path d="M16 7.5A6.25 6.25 0 1 0 16.2 11M16 4v3.75h-3.75" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/></svg>; }
function TrashIcon() { return <svg viewBox="0 0 20 20" fill="none" aria-hidden="true"><path d="M4.5 6h11m-9.75 0 .6 9.25c.05.7.63 1.25 1.33 1.25h4.64c.7 0 1.28-.55 1.33-1.25L15.25 6M8 6V4.75c0-.41.34-.75.75-.75h2.5c.41 0 .75.34.75.75V6m-3 3v4.5m3-4.5v4.5" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round"/></svg>; }
function LockIcon() { return <svg viewBox="0 0 18 18" fill="none" aria-hidden="true"><rect x="3.25" y="7.5" width="11.5" height="8.25" rx="1.5" stroke="currentColor" strokeWidth="1.3"/><path d="M5.75 7.5V5.25a3.25 3.25 0 0 1 6.5 0V7.5M9 10.5v2.25" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round"/></svg>; }
function SparkleIcon() { return <svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="m12 3 1.85 5.15L19 10l-5.15 1.85L12 17l-1.85-5.15L5 10l5.15-1.85L12 3ZM19 15l.85 2.15L22 18l-2.15.85L19 21l-.85-2.15L16 18l2.15-.85L19 15Z" stroke="currentColor" strokeWidth="1.3" strokeLinejoin="round"/></svg>; }

createRoot(document.getElementById("root")).render(<App />);
