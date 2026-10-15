"""Question-aware ranking over a Pinecone candidate set."""

import math
import re
import unicodedata

from app.config import settings
from app.models import Candidate, Chunk

_TOKEN = re.compile(r"[a-z0-9]+|[\u0900-\u097f]+", re.IGNORECASE)
_STOP = set("""a an the and or but is are was were be been being do does did have has had
how what when where which who whom why many much tell me about of in on at to for from with
by as it this that these those my your their his her its i we they he she you please give find
hai hain tha thi the kya kab kahan kitna kitne kitni ka ki ke mein me par se ko aur batao bata
mera meri mere us uska uski uske unka unki hain hai tha thi""".split())


def terms(text: str) -> list[str]:
    words = [unicodedata.normalize("NFKC", item).casefold() for item in _TOKEN.findall(text)]
    return [word[:-1] if len(word) > 4 and word.endswith("s") else word
            for word in words if word not in _STOP and (len(word) > 1 or word.isdigit())]


def lexical_match(question: str, text: str) -> float:
    query_terms = set(terms(question))
    if not query_terms:
        return 0.0
    body_terms = set(terms(text))
    exact = len(query_terms & body_terms)
    # Keep lexical contribution bounded; semantic similarity remains the first-stage gate.
    return exact / len(query_terms)


def rank_candidates(question: str, rows: list[tuple[Chunk, float]], limit: int | None = None,
                     min_similarity: float | None = None) -> list[Candidate]:
    floor = settings.min_similarity if min_similarity is None else min_similarity
    ranked = []
    for chunk, raw_score in rows:
        score = float(raw_score or 0.0)
        if score < floor:
            continue
        lexical = lexical_match(question, chunk.text)
        # Exact query terminology, especially names/count labels, breaks ties between semantically
        # similar passages such as a connection count and a follower count.
        combined = score + 0.12 * lexical
        ranked.append(Candidate(chunk, score, lexical, combined))
    ranked.sort(key=lambda item: (item.rank_score, item.lexical_score, item.similarity), reverse=True)
    if limit is None:
        limit = settings.context_k
    return ranked[:limit]


def _focused_source_text(question: str, candidates: list[Candidate]) -> list[tuple[Candidate, str]]:
    question_folded = unicodedata.normalize("NFKC", question).casefold()
    employment_question = bool(re.search(
        r"\b(work|working|employer|employment|job|position|role|experience)\b|काम|नौकरी|kaam|naukri",
        question_folded))
    skills_question = bool(re.search(r"\b(skill|skills|technical|technology|technologies)\b|कौशल|skills?",
                                     question_folded))
    focused: list[tuple[Candidate, str]] = []
    if employment_question:
        for item in candidates:
            for match in re.finditer(r"\bEXPERIENCE\b\s*([^\x7f]{1,350})", item.chunk.text):
                excerpt = " ".join(match.group(1).split()).strip(" :-|\n")
                if "|" in excerpt and re.search(r"\b(?:19|20)\d{2}\b|present|current", excerpt, re.I):
                    focused.append((item, excerpt))
        current = [row for row in focused if re.search(r"\bpresent\b|\bcurrent\b", row[1], re.I)]
        if current and not re.search(r"\b(all|every|previous|past|history)\b", question_folded):
            return current[:2]
        if focused:
            return focused[:4]
    if skills_question:
        for item in candidates:
            match = re.search(
                r"\bTECHNICAL SKILLS\b\s*(.*?)(?=\bEXPERIENCE\b|\bPROJECTS\b|\bEDUCATION\b|$)",
                item.chunk.text, re.DOTALL)
            if match:
                excerpt = " ".join(match.group(1).split()).strip(" :-|\n")
                if excerpt:
                    focused.append((item, excerpt))
        if focused:
            return focused[:2]
    return []


def prepare_context(candidates: list[Candidate], question: str | None = None) -> str:
    focused = _focused_source_text(question, candidates) if question else []
    source_rows = focused or [(item, item.chunk.text) for item in candidates]
    excerpts = []
    seen = set()
    for number, (item, source_text) in enumerate(source_rows, 1):
        text = " ".join(source_text.split())
        key = re.sub(r"\W+", " ", text.casefold()).strip()
        if not text or key in seen:
            continue
        seen.add(key)
        meta = item.chunk.metadata
        locators = []
        for key_name, label in (("page", "page"), ("slide", "slide"), ("sheet", "sheet")):
            if meta.get(key_name):
                locators.append(f"{label} {meta[key_name]}")
        location = ", ".join(locators)
        name = str(meta.get("filename") or meta.get("source") or "Document")
        label = f"{name} ({location})" if location else name
        excerpts.append(f"[E{number} | {label}]\n{text}")
    return "\n\n".join(excerpts)
