"""Question-aware ranking over a Pinecone candidate set.

Pinecone returns the semantically closest chunks. Here they are re-ranked with a small bonus
for exact question terms, then turned into the labelled excerpts shown to the model.
Nothing in this file depends on the document type or topic.
"""

import re
import unicodedata

from app.config import settings
from app.models import Candidate, Chunk

# How much exact term overlap can add to the semantic score (overlap is 0..1).
LEXICAL_WEIGHT = 0.12

_TOKEN = re.compile(r"[a-z0-9]+|[\u0900-\u097f]+", re.IGNORECASE)
# Function words only (English plus common Hinglish question words); no topic words.
_STOP = set("""a an the and or but is are was were be been being do does did have has had
how what when where which who whom why many much tell me about of in on at to for from with
by as it this that these those my your their his her its i we they he she you please give find
hai hain tha thi kya kab kahan kitna kitne kitni ka ki ke mein me par se ko aur batao bata
mera meri mere us uska uski uske unka unki""".split())


def terms(text: str) -> list[str]:
    """Lower-cased content words of `text`, without stop words and with a trailing plural 's' cut."""
    words = [unicodedata.normalize("NFKC", item).casefold() for item in _TOKEN.findall(text)]
    return [word[:-1] if len(word) > 4 and word.endswith("s") else word
            for word in words if word not in _STOP and (len(word) > 1 or word.isdigit())]


def lexical_match(question: str, text: str) -> float:
    """Share (0..1) of the question's content words that also appear in `text`."""
    query_terms = set(terms(question))
    if not query_terms:
        return 0.0
    return len(query_terms & set(terms(text))) / len(query_terms)


def rank_candidates(question: str, rows: list[tuple[Chunk, float]], limit: int | None = None,
                    min_similarity: float | None = None) -> list[Candidate]:
    """Drop weak semantic matches, then order the rest by similarity plus an exact-term bonus."""
    floor = settings.min_similarity if min_similarity is None else min_similarity
    ranked = []
    for chunk, raw_score in rows:
        score = float(raw_score or 0.0)
        if score < floor:
            continue
        # Exact wording breaks ties between passages that are semantically close.
        lexical = lexical_match(question, chunk.text)
        ranked.append(Candidate(chunk, score, lexical, score + LEXICAL_WEIGHT * lexical))
    ranked.sort(key=lambda item: (item.rank_score, item.lexical_score, item.similarity), reverse=True)
    return ranked[:settings.context_k if limit is None else limit]


def _source_label(metadata: dict) -> str:
    name = str(metadata.get("filename") or metadata.get("source") or "Document")
    places = [f"{label} {metadata[key]}" for key, label in
              (("page", "page"), ("slide", "slide"), ("sheet", "sheet")) if metadata.get(key)]
    return f"{name} ({', '.join(places)})" if places else name


def prepare_context(candidates: list[Candidate], question: str | None = None) -> str:
    """Join the ranked chunks into numbered, source-labelled excerpts, skipping duplicates.

    `question` is accepted so existing callers keep working; the context does not depend on it.
    """
    excerpts: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        text = " ".join(item.chunk.text.split())
        key = re.sub(r"\W+", " ", text.casefold()).strip()
        if not text or key in seen:
            continue
        seen.add(key)
        excerpts.append(f"[E{len(excerpts) + 1} | {_source_label(item.chunk.metadata)}]\n{text}")
    return "\n\n".join(excerpts)