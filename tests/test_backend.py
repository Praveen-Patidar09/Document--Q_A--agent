import io
import json
import unittest
from unittest import mock

from fastapi import Response, UploadFile

from app import main
from app.answerer import NOT_FOUND, validate_response
from app.chunking import make_chunks, split_text
from app.models import Candidate, Chunk, DocumentPart
from app.parsers import extract_document
from app.pinecone_store import PineconeDocumentStore
from app.retrieval import prepare_context, rank_candidates
from app.schemas import QuestionRequest


def response(answer, evidence):
    return json.dumps({"answer": answer, "evidence": evidence}, ensure_ascii=False)


class ParserAndChunkTests(unittest.TestCase):
    def test_text_normalizes_unicode_and_keeps_paragraphs(self):
        parts = extract_document("memo.txt", "Ａmount: 500\r\n\r\nDate: 12 May 2026".encode())
        self.assertEqual(parts[0].text, "Amount: 500\n\nDate: 12 May 2026")

    def test_unsupported_and_empty_uploads_are_rejected(self):
        with self.assertRaises(ValueError):
            extract_document("notes.exe", b"abc")
        with self.assertRaises(ValueError):
            extract_document("notes.txt", b"")

    def test_docx_pptx_and_xlsx_keep_useful_locations(self):
        from docx import Document as WordDocument
        from openpyxl import Workbook
        from pptx import Presentation
        from pptx.util import Inches

        word = WordDocument()
        word.add_heading("Experience", level=1)
        word.add_paragraph("Worked at Northwind.")
        word_bytes = io.BytesIO()
        word.save(word_bytes)
        docx_parts = extract_document("cv.docx", word_bytes.getvalue())
        self.assertEqual(docx_parts[0].metadata["heading"], "Experience")
        self.assertIn("Northwind", docx_parts[0].text)

        deck = Presentation()
        slide = deck.slides.add_slide(deck.slide_layouts[6])
        box = slide.shapes.add_textbox(Inches(1), Inches(1), Inches(5), Inches(1))
        box.text = "Launch date: 12 May 2026"
        deck_bytes = io.BytesIO()
        deck.save(deck_bytes)
        ppt_parts = extract_document("plan.pptx", deck_bytes.getvalue())
        self.assertEqual(ppt_parts[0].metadata["slide"], 1)

        book = Workbook()
        sheet = book.active
        sheet.title = "Milestones"
        sheet.append(["Name", "Date"])
        sheet.append(["Launch", "12 May 2026"])
        book_bytes = io.BytesIO()
        book.save(book_bytes)
        xlsx_parts = extract_document("plan.xlsx", book_bytes.getvalue())
        self.assertEqual(xlsx_parts[0].metadata["sheet"], "Milestones")
        self.assertIn("12 May 2026", xlsx_parts[0].text)

    def test_chunks_keep_page_metadata_and_boundaries(self):
        parts = [DocumentPart("\n\n".join(["Page one has 359 connections."] * 30), {"page": 1}),
                 DocumentPart("Page two has another fact.", {"page": 2})]
        chunks = make_chunks(parts, "profile.pdf", "a" * 24)
        self.assertTrue(chunks)
        self.assertEqual(chunks[0].metadata["page"], 1)
        self.assertEqual(chunks[-1].metadata["page"], 2)
        self.assertEqual([chunk.metadata["chunk"] for chunk in chunks], list(range(len(chunks))))
        self.assertTrue(all(len(chunk.text) <= 1100 for chunk in chunks))

    def test_chunking_does_not_join_unrelated_paragraphs_when_under_limit(self):
        parts = [DocumentPart("359 LinkedIn connections on page one."),
                 DocumentPart("4.2 million followers on another page.")]
        chunks = make_chunks(parts, "profile.pdf", "b" * 24)
        self.assertEqual(len(chunks), 2)
        self.assertNotIn("followers", chunks[0].text)


class RetrievalTests(unittest.TestCase):
    def test_linkedin_connections_rank_ahead_of_follower_count(self):
        rows = [
            (Chunk("The profile has 4,200,000 followers.", {"page": 8, "chunk": 8}), 0.81),
            (Chunk("LinkedIn connections on page one: 359.", {"page": 1, "chunk": 1}), 0.77),
        ]
        ranked = rank_candidates("How many LinkedIn connections are listed on page 1?", rows,
                                 limit=2, min_similarity=0.4)
        self.assertEqual(ranked[0].chunk.metadata["page"], 1)
        self.assertIn("359", ranked[0].chunk.text)

    def test_similarity_threshold_is_applied_before_lexical_reranking(self):
        rows = [(Chunk("Connections: 359"), 0.39), (Chunk("Connections: 412"), 0.55)]
        result = rank_candidates("How many connections?", rows, min_similarity=0.42)
        self.assertEqual(len(result), 1)
        self.assertIn("412", result[0].chunk.text)

    def test_context_contains_only_ranked_chunks_and_source_labels(self):
        items = [Candidate(Chunk("Connections: 359", {"filename": "p.pdf", "page": 1}),
                           0.9, 0.5, 0.96)]
        context = prepare_context(items)
        self.assertIn("p.pdf (page 1)", context)
        self.assertIn("Connections: 359", context)
        self.assertNotIn("follower", context)

    def test_context_focuses_resume_work_and_skill_sections(self):
        text = ("SUMMARY AI intern. TECHNICAL SKILLS Python, FastAPI, and Git. "
                "EXPERIENCE AI Intern | SourceBae | Aug 2026 - Present \x7f Building AI projects. "
                "Software Engineer Intern | OtherCo | 2024 - 2025")
        item = Candidate(Chunk(text, {"filename": "resume.pdf", "page": 1}), 0.9, 0.0, 0.9)
        work_context = prepare_context([item], "Where does Praveen currently work?")
        self.assertIn("AI Intern | SourceBae | Aug 2026 - Present", work_context)
        self.assertNotIn("OtherCo", work_context)
        self.assertNotIn("Building AI projects", work_context)
        skill_context = prepare_context([item], "Summarize the technical skills.")
        self.assertIn("Python, FastAPI, and Git", skill_context)
        self.assertNotIn("SourceBae", skill_context)


class AnswerValidationTests(unittest.TestCase):
    CONTEXT = "[E1 | profile.pdf (page 1)]\nLinkedIn connections: 359 on page one."

    def test_answer_with_exact_quote_and_number_is_accepted(self):
        raw = response("The profile lists 359 LinkedIn connections.", ["LinkedIn connections: 359 on page one."])
        self.assertEqual(validate_response(raw, "How many connections?", self.CONTEXT),
                         "The profile lists 359 LinkedIn connections.")

    def test_supported_job_paraphrase_does_not_get_rejected_for_question_name_or_month_spelling(self):
        context = "[E1 | resume.pdf]\nAI Engineer Intern — Sourcebae — Aug 2026 to Present."
        answer = "Praveen currently works as an AI Engineer Intern at Sourcebae, starting in August 2026."
        raw = response(answer, ["AI Engineer Intern — Sourcebae — Aug 2026 to Present."])
        self.assertEqual(validate_response(raw, "What is Praveen's current job?", context), answer)

    def test_linkedin_regression_rejects_unrelated_follower_number(self):
        raw = response("The profile has 4,200,000 connections.", ["LinkedIn connections: 359 on page one."])
        self.assertEqual(validate_response(raw, "How many LinkedIn connections?", self.CONTEXT), NOT_FOUND)

    def test_answer_cannot_swap_metric_or_person_name(self):
        self.assertEqual(validate_response(response("There are 359 followers.",
                                                    ["LinkedIn connections: 359 on page one."]),
                                           "How many connections?", self.CONTEXT), NOT_FOUND)
        people_context = "[E1] Priya Shah scored 91 points. Priya Sen scored 78 points."
        self.assertEqual(validate_response(response("Priya Sen scored 91 points.",
                                                    ["Priya Shah scored 91 points."]),
                                           "What did Priya Shah score?", people_context), NOT_FOUND)

    def test_person_and_number_must_be_supported_in_same_evidence_quote(self):
        people_context = "[E1] Priya Shah scored 91 points. Priya Sen scored 78 points."
        raw = response("Priya Sen scored 91 points.", ["Priya Shah scored 91 points."])
        self.assertEqual(validate_response(raw, "What did Priya Shah score?", people_context), NOT_FOUND)

    def test_common_word_like_does_not_trigger_metric_guard_but_counted_likes_do(self):
        skills_context = "[E1] Tools: Git, GitHub, VS Code, and Postman."
        skills_answer = response("Tools like Git and GitHub.",
                                 ["Tools: Git, GitHub, VS Code, and Postman."])
        self.assertEqual(validate_response(skills_answer, "Summarize the skills.", skills_context),
                         "Tools like Git and GitHub.")
        count_context = "[E1] LinkedIn connections: 359."
        swapped = response("359 likes.", ["LinkedIn connections: 359."])
        self.assertEqual(validate_response(swapped, "How many connections?", count_context), NOT_FOUND)

    def test_answer_cannot_add_an_unsupported_qualifier(self):
        raw = response("The profile is verified and has 359 connections.",
                       ["LinkedIn connections: 359 on page one."])
        self.assertEqual(validate_response(raw, "How many connections?", self.CONTEXT), NOT_FOUND)

    def test_answer_cannot_use_evidence_from_outside_selected_context(self):
        raw = response("There are 359 connections.", ["There are 359 connections."])
        self.assertEqual(validate_response(raw, "How many?", "Another selected document says 12."), NOT_FOUND)

    def test_missing_answer_returns_exact_fixed_refusal(self):
        raw = response(NOT_FOUND, [])
        self.assertEqual(validate_response(raw, "What is the address?", self.CONTEXT), NOT_FOUND)

    def test_partial_answer_is_kept_and_missing_part_is_clear(self):
        answer = "The profile lists 359 connections. Its creation date is not provided."
        raw = response(answer, ["LinkedIn connections: 359 on page one."])
        self.assertEqual(validate_response(raw, "How many connections and when was it created?", self.CONTEXT), answer)

    def test_duplicate_refusal_is_removed_from_grounded_answer(self):
        answer = f"There are 359 connections. {NOT_FOUND}"
        raw = response(answer, ["LinkedIn connections: 359 on page one."])
        self.assertEqual(validate_response(raw, "How many?", self.CONTEXT), "There are 359 connections.")

    def test_malformed_truncated_wrong_schema_and_fake_evidence_fail_closed(self):
        for raw in ("{truncated", '{"answer": 4, "evidence": []}',
                    '{"answer":"359 connections","evidence":["not in doc"]}',
                    '{"answer":"359 connections","evidence":[],"extra":1}'):
            with self.subTest(raw=raw):
                self.assertEqual(validate_response(raw, "How many?", self.CONTEXT), NOT_FOUND)
        self.assertEqual(validate_response(None, "How many?", self.CONTEXT), NOT_FOUND)

    def test_hindi_model_output_falls_back_to_short_verified_english_quote(self):
        raw = response("18 दिन की छुट्टी मिलती है", ["Employees receive 18 days of annual leave."])
        context = "[E1 | policy.pdf]\nEmployees receive 18 days of annual leave."
        self.assertEqual(validate_response(raw, "कितने दिन की छुट्टी मिलती है?", context),
                         "Employees receive 18 days of annual leave.")

    def test_supported_fact_refusal_gets_one_focused_recheck(self):
        from app.answerer import OllamaAnswerer

        class FakeResponse:
            def __init__(self, payload): self.payload = payload
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self): return json.dumps({"message": {"content": self.payload},
                                               "done_reason": "stop"}).encode()

        adapter = OllamaAnswerer()
        first = response(NOT_FOUND, [])
        second = response("The annual fee is $240. A renewal date is not specified.",
                          ["The annual fee is $240."])
        with mock.patch.object(adapter.opener, "open", side_effect=[FakeResponse(first),
                                                                     FakeResponse(second)]) as calls:
            result = adapter.answer("What is the fee and renewal date?",
                                    "[E1] The annual fee is $240. Accounts can be renewed online.")
        self.assertEqual(result, "The annual fee is $240. A renewal date is not specified.")
        self.assertEqual(calls.call_count, 2)

    def test_ollama_adapter_makes_one_structured_call_with_selected_context(self):
        from app.answerer import OllamaAnswerer

        raw = response("There are 359 connections.", ["LinkedIn connections: 359 on page one."])

        class FakeResponse:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self): return json.dumps({"message": {"content": raw}, "done_reason": "stop"}).encode()

        adapter = OllamaAnswerer()
        with mock.patch.object(adapter.opener, "open", return_value=FakeResponse()) as open_call:
            answer = adapter.answer("How many connections?", self.CONTEXT)
        self.assertEqual(answer, "There are 359 connections.")
        self.assertEqual(open_call.call_count, 1)
        request = open_call.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], main.settings.ollama_model)
        self.assertEqual(payload["format"], __import__("app.answerer", fromlist=["OUTPUT_SCHEMA"]).OUTPUT_SCHEMA)
        self.assertIn(self.CONTEXT, payload["messages"][1]["content"])


class EmbeddingTests(unittest.TestCase):
    def test_gemini_embedding_dimensions_and_task_types_are_checked(self):
        from app import embeddings
        from app.config import settings

        class Models:
            def __init__(self): self.calls = []
            def embed_content(self, **kwargs):
                self.calls.append(kwargs)
                contents = kwargs["contents"]
                count = len(contents) if isinstance(contents, list) else 1
                return type("Result", (), {"embeddings": [type("E", (), {"values": [0.1] * settings.embedding_dimension})()
                                                              for _ in range(count)]})()

        client = type("Client", (), {})()
        client.models = Models()
        adapter = embeddings.GeminiEmbeddings.__new__(embeddings.GeminiEmbeddings)
        adapter.client = client
        with mock.patch.object(embeddings, "_retry", side_effect=lambda call: call()):
            vectors = adapter.documents(["first", "second"])
            query = adapter.query("question")
        self.assertEqual(len(vectors), 2)
        self.assertEqual(len(query), settings.embedding_dimension)
        self.assertEqual(client.models.calls[0]["config"].task_type, "RETRIEVAL_DOCUMENT")
        self.assertEqual(client.models.calls[1]["config"].task_type, "RETRIEVAL_QUERY")


class AnswerLanguageTests(unittest.TestCase):
    CONTEXT = "[E1 | profile.pdf (page 1)]\nLinkedIn connections: 359 on page one."

    def test_escaped_newline_is_repaired_and_devanagari_output_is_rejected(self):
        raw = response("359 connections\\nPage one.", ["LinkedIn connections: 359 on page one."])
        self.assertEqual(validate_response(raw, "How many?", self.CONTEXT), "359 connections\nPage one.")
        hindi = response("359 कनेक्शन", ["LinkedIn connections: 359 on page one."])
        self.assertEqual(validate_response(hindi, "कितने connections?", self.CONTEXT),
                         "LinkedIn connections: 359 on page one.")
        hinglish = response("359 connections hain.", ["LinkedIn connections: 359 on page one."])
        self.assertEqual(validate_response(hinglish, "kitne connections?", self.CONTEXT),
                         "LinkedIn connections: 359 on page one.")

    def test_english_hindi_and_hinglish_questions_accept_english_answer(self):
        raw = response("The connection count is 359.", ["LinkedIn connections: 359 on page one."])
        for question in ("How many connections?", "कितने connections हैं?", "Connections kitne hain?"):
            with self.subTest(question=question):
                self.assertEqual(validate_response(raw, question, self.CONTEXT), "The connection count is 359.")


class FakeIndex:
    def __init__(self):
        self.namespaces = {}
        self.last_query_namespace = None

    def upsert(self, vectors, namespace):
        self.namespaces.setdefault(namespace, {}).update({item["id"]: item for item in vectors})

    def fetch(self, ids, namespace):
        values = self.namespaces.get(namespace, {})
        return {"vectors": {key: values[key] for key in ids if key in values}}

    def describe_index_stats(self):
        return {"dimension": 768, "namespaces": {
            namespace: {"vector_count": len(vectors)} for namespace, vectors in self.namespaces.items()}}

    def query(self, vector, top_k, namespace, include_metadata):
        self.last_query_namespace = namespace
        values = list(self.namespaces.get(namespace, {}).values())
        return {"matches": [{"metadata": item["metadata"], "score": 0.9}
                            for item in values if item["metadata"].get("record_type") == "chunk"][:top_k]}

    def delete(self, delete_all, namespace):
        self.namespaces.pop(namespace, None)

    def list(self, namespace):
        return [{"vectors": list(self.namespaces.get(namespace, {}))}]


class StoreTests(unittest.TestCase):
    def test_index_verify_namespace_query_list_and_delete(self):
        index = FakeIndex()
        store = PineconeDocumentStore(index=index)
        chunks = [Chunk("Connections: 359", {"document_id": "a" * 24, "filename": "profile.pdf", "chunk": 0})]
        store.add_document("a" * 24, "profile.pdf", chunks, [[0.0] * 768], "2026-10-14T00:00:00Z")
        self.assertTrue(store.document_exists("a" * 24))
        rows = store.search("a" * 24, [0.0] * 768, 10)
        self.assertEqual(len(rows), 1)
        self.assertEqual(index.last_query_namespace, "a" * 24)
        self.assertEqual(store.list_documents()[0]["filename"], "profile.pdf")
        self.assertTrue(store.delete_document("a" * 24))
        self.assertFalse(store.document_exists("a" * 24))

    def test_incomplete_manifest_is_not_ready(self):
        index = FakeIndex()
        store = PineconeDocumentStore(index=index)
        doc_id = "c" * 24
        store.add_document(doc_id, "incomplete.txt",
                           [Chunk("One", {"document_id": doc_id, "filename": "incomplete.txt"})],
                           [[0.0] * 768], "2026-10-14T00:00:00Z")
        index.namespaces[doc_id].pop(f"{doc_id}-c000000")
        self.assertFalse(store.document_exists(doc_id))

    def test_two_document_namespaces_do_not_cross_search(self):
        index = FakeIndex()
        store = PineconeDocumentStore(index=index)
        for doc_id, name, text in (("a" * 24, "a.txt", "A facts"), ("b" * 24, "b.txt", "B facts")):
            store.add_document(doc_id, name, [Chunk(text, {"document_id": doc_id, "filename": name})],
                               [[0.0] * 768], "2026-10-14T00:00:00Z")
        rows = store.search("a" * 24, [0.0] * 768, 10)
        self.assertEqual([row[0].metadata["filename"] for row in rows], ["a.txt"])


class FakeMemoryStore:
    def __init__(self):
        self.docs = {}
        self.score = 0.9
        self.searched = []

    def document_exists(self, doc_id):
        return doc_id in self.docs

    def add_document(self, doc_id, filename, chunks, vectors, created):
        self.docs[doc_id] = (filename, chunks)

    def search(self, doc_id, vector, top_k):
        self.searched.append(doc_id)
        return [(chunk, self.score) for chunk in self.docs[doc_id][1]]

    def list_documents(self):
        return [{"document_id": key, "filename": val[0], "chunks": len(val[1]), "created_at": None}
                for key, val in self.docs.items()]

    def delete_document(self, doc_id):
        return self.docs.pop(doc_id, None) is not None


class FakeEmbeddings:
    def documents(self, texts):
        return [[0.1] * 768 for _ in texts]

    def query(self, question):
        return [0.1] * 768


class FakeAnswerer:
    def answer(self, question, context):
        return "The profile lists 359 LinkedIn connections."


class FrontendContractAndIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.old = main.get_store, main.get_embeddings, main.get_answerer
        self.store = FakeMemoryStore()
        main.get_store = lambda: self.store
        main.get_embeddings = lambda: FakeEmbeddings()
        main.get_answerer = lambda: FakeAnswerer()

    def tearDown(self):
        main.get_store, main.get_embeddings, main.get_answerer = self.old

    def test_frontend_routes_and_upload_ask_response_contract(self):
        paths = {(route.path, tuple(sorted(route.methods or []))) for route in main.app.routes}
        self.assertIn(("/health", ("GET",)), paths)
        self.assertIn(("/documents", ("GET",)), paths)
        self.assertTrue(any(path == "/documents" and "POST" in methods for path, methods in paths))
        self.assertTrue(any(path == "/questions" and "POST" in methods for path, methods in paths))
        response = Response()
        upload = main.upload_document(response, UploadFile(filename="profile.txt",
                                      file=io.BytesIO(b"LinkedIn connections: 359 on page one.")))
        result = main.ask_question(QuestionRequest(question="How many LinkedIn connections?",
                                                    document_id=upload["document_id"]))
        self.assertEqual(set(result), {"answer", "sources"})
        self.assertEqual(result["answer"], "The profile lists 359 LinkedIn connections.")
        self.assertEqual(result["sources"][0]["filename"], "profile.txt")
        self.assertEqual(self.store.searched, [upload["document_id"]])

    def test_unavailable_retrieval_abstains_and_does_not_call_answerer(self):
        class NoAnswer:
            def answer(self, question, context):
                raise AssertionError("Answerer must not run when retrieval is below threshold.")
        main.get_answerer = lambda: NoAnswer()
        self.store.score = 0.1
        response = Response()
        upload = main.upload_document(response, UploadFile(filename="profile.txt",
                                      file=io.BytesIO(b"LinkedIn connections: 359.")))
        result = main.ask_question(QuestionRequest(question="What is the address?",
                                                    document_id=upload["document_id"]))
        self.assertEqual(result, {"answer": NOT_FOUND, "sources": []})

    def test_missing_selected_document_is_a_404(self):
        with self.assertRaises(Exception) as caught:
            main.ask_question(QuestionRequest(question="Question", document_id="f" * 24))
        self.assertEqual(caught.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
