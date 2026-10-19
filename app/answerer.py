"""Ollama answer generation plus structural validation of the model's reply.

The validator only checks things that can be verified without understanding language:

  1. the reply has the expected JSON shape,
  2. every evidence quote really comes from the retrieved excerpts,
  3. every number in the answer appears in those quotes,
  4. a person's name and a number stay together in one quote,
  5. the answer is English (Devanagari / Hinglish falls back to the verbatim quote).

There are no word lists and no topic-specific rules, so it behaves the same for any
document. Checks that need real language understanding (synonyms, paraphrase, label swaps)
are deliberately NOT done here with word matching; they belong in a meaning-based check.
"""

import json
import logging
import re
import time
import unicodedata
import urllib.error
import urllib.request

from app.config import settings
from app.diagnostics import trace
from app.prompts import SYSTEM_PROMPT, user_prompt
from app.retrieval import terms

logger = logging.getLogger(__name__)

NOT_FOUND = "This information is not available in the selected document."
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}, "evidence": {"type": "array", "items": {"type": "string"}}},
    "required": ["answer", "evidence"],
    "additionalProperties": False,
}

_NUMBER = re.compile(r"(?<!\w)(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?(?!\w)")
_HINDI = re.compile(r"[ऀ-ॿ]")
_WORD = re.compile(r"\w+")
_REFUSAL = re.compile(re.escape(NOT_FOUND), re.IGNORECASE)
_PERSON_NAME = re.compile(r"\b([A-Z][a-z]{2,})\s+([A-Z][a-z]{2,})\b")
# Latin-script Hindi words that signal the model ignored the English-only instruction.
_HINGLISH_MARKERS = {"hai", "hain", "tha", "thi", "hoga", "hogi", "honge", "nahi", "nahin",
                     "kya", "kab", "kaise", "kitna", "kitni", "kitne", "kyunki", "mein", "se", "ko",
                     "par", "aur", "bhi", "wala", "wali", "wale", "mila", "milta", "milti"}

_MAX_QUOTE_CHARS = 1200
_MAX_FALLBACK_QUOTE_CHARS = 240
_MIN_FUZZY_QUOTE_WORDS = 4   # shorter quotes must match exactly
_FUZZY_WINDOW_FACTOR = 2     # matched words may spread over at most 2x the quote length (+4)
_QUOTE_CHARS = "\"'“”‘’ \t\r\n"
_TRIM_CHARS = " \t\r\n-–—;,"


# --------------------------------------------------------------------------- text helpers

def _normalize_for_match(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _unescape_output(text: str) -> str:
    # JSON decodes ordinary escapes. Some local models double-escape line breaks; repair those.
    return text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", " ").strip()


def _strip_wrapping_quotes(text: str) -> str:
    """Models often wrap a quote in their own quotation marks; those are not document text."""
    return text.strip(_QUOTE_CHARS)


def _numbers(text: str) -> set[str]:
    return {number.replace(",", "") for number in _NUMBER.findall(text)}


def _name_pairs(text: str) -> set[str]:
    return {" ".join(match).casefold() for match in _PERSON_NAME.findall(text)}


def _word_tokens(text: str) -> list[tuple[str, int, int]]:
    """(normalized word, start, end) for every word; offsets point into the original text."""
    return [(unicodedata.normalize("NFKC", m.group()).casefold(), m.start(), m.end())
            for m in _WORD.finditer(text)]


# --------------------------------------------------------------------------- quote grounding

def _span_in_block(quote_words: list[str], block: str) -> str | None:
    """Return the verbatim text of `block` that contains the quote's words in order.

    This tolerates a model shortening a quote (dropping words in the middle) while still
    requiring every quoted word to exist, in order, close together, in the same excerpt.
    """
    words = _word_tokens(block)
    window = len(quote_words) * _FUZZY_WINDOW_FACTOR + 4
    for start, (word, first_char, _) in enumerate(words):
        if word != quote_words[0]:
            continue
        matched, last = 1, start
        for position in range(start + 1, min(len(words), start + window)):
            if matched == len(quote_words):
                break
            if words[position][0] == quote_words[matched]:
                matched += 1
                last = position
        if matched == len(quote_words):
            return block[first_char:words[last][2]]
    return None


def _ground_quote(quote: str, normalized_context: str, blocks: list[str]) -> str | None:
    """Return text that is verbatim from the context, or None if the quote is not supported."""
    quote = _strip_wrapping_quotes(_unescape_output(quote))
    if not quote or len(quote) > _MAX_QUOTE_CHARS:
        return None
    if _normalize_for_match(quote) in normalized_context:
        return quote
    quote_words = [word for word, _, _ in _word_tokens(quote)]
    if len(quote_words) < _MIN_FUZZY_QUOTE_WORDS:
        return None
    for block in blocks:
        span = _span_in_block(quote_words, block)
        if span and len(span) <= _MAX_QUOTE_CHARS:
            return span
    return None


def _grounded_quotes(evidence: list[str], context: str) -> list[str]:
    normalized_context = _normalize_for_match(context)
    blocks = [block for block in re.split(r"\n\s*\n", context) if block.strip()]
    grounded = (_ground_quote(item, normalized_context, blocks) for item in evidence if item.strip())
    return [quote for quote in grounded if quote]


# --------------------------------------------------------------------------- validation

def parse_json(raw: str) -> dict | None:
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if (not isinstance(value, dict) or set(value) != {"answer", "evidence"}
            or not isinstance(value["answer"], str)
            or not isinstance(value["evidence"], list)
            or any(not isinstance(item, str) for item in value["evidence"])):
        return None
    return value


def _person_and_number_share_a_quote(answer: str, question: str, quotes: list[str]) -> bool:
    """Stop a number for one person being given to a similarly named person."""
    answer_numbers = _numbers(answer)
    people = _name_pairs(answer) - _name_pairs(question)
    if not (_name_pairs(question) and people and answer_numbers):
        return True
    return any(
        all(person in _normalize_for_match(quote) for person in people)
        and answer_numbers.issubset(_numbers(quote))
        for quote in quotes
    )


def _is_not_english(answer: str) -> bool:
    return bool(_HINDI.search(answer)
                or set(re.findall(r"[a-z]+", answer.casefold())) & _HINGLISH_MARKERS)


def validate_response(raw: str, question: str, context: str) -> str:
    """Return a validated answer, or NOT_FOUND when the reply cannot be trusted."""
    result = parse_json(raw)
    if result is None:
        logger.warning("Ollama response was malformed or truncated; returning fixed abstention.")
        return NOT_FOUND

    answer = _unescape_output(result["answer"])
    if _REFUSAL.fullmatch(answer):
        return NOT_FOUND
    # Remove an exact contradictory refusal before validating the supported answer itself.
    answer = _REFUSAL.sub("", answer).strip(_TRIM_CHARS)
    if not answer or not any(item.strip() for item in result["evidence"]):
        return NOT_FOUND

    quotes = _grounded_quotes(result["evidence"], context)
    if not quotes:
        logger.warning("Ollama answer had no evidence quote matching retrieved text.")
        return NOT_FOUND

    # A real quote is necessary but not sufficient: every number asserted must be in the quotes.
    if not _numbers(answer).issubset(_numbers("\n".join(quotes))):
        logger.warning("Ollama answer introduced a number absent from its evidence quotes.")
        return NOT_FOUND

    if not _person_and_number_share_a_quote(answer, question, quotes):
        logger.warning("Answer combined a different person's name with the requested number.")
        return NOT_FOUND

    # English is required. If the model ignored that, a short verbatim English quote is safer
    # than refusing a fact only because the model answered in Hindi.
    if _is_not_english(answer):
        logger.warning("Ollama ignored the English-only answer requirement.")
        quote = min(quotes, key=len)
        if len(quote) <= _MAX_FALLBACK_QUOTE_CHARS and not _HINDI.search(quote):
            return quote
        return NOT_FOUND

    answer = re.sub(r"[ \t]{2,}", " ", answer)
    answer = re.sub(r"\n{3,}", "\n\n", answer).strip(_TRIM_CHARS)
    return answer or NOT_FOUND


# --------------------------------------------------------------------------- Ollama client

class OllamaAnswerer:
    def __init__(self):
        self.base_url = settings.ollama_base_url
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def installed_models(self) -> list[str]:
        with self.opener.open(f"{self.base_url}/api/tags", timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return [item.get("name", "") for item in payload.get("models", [])]

    def check(self) -> tuple[bool, bool]:
        try:
            models = self.installed_models()
        except (OSError, urllib.error.URLError, ValueError):
            return False, False
        wanted = settings.ollama_model
        return True, wanted in models or f"{wanted}:latest" in models

    @staticmethod
    def _needs_recheck(question: str, context: str) -> bool:
        """After a bare refusal, retry once if the question clearly overlaps the excerpts."""
        conjunction = re.search(r"\b(and|or|also|aur|bhi)\b|और|तथा", question, re.IGNORECASE)
        overlap = set(terms(question)) & set(terms(context))
        return len(overlap) >= 2 or (bool(conjunction) and bool(overlap))

    def _generate(self, question: str, context: str, retry_partial: bool = False) -> tuple[str, str | None]:
        prompt = user_prompt(question, context)
        if retry_partial:
            prompt += ("\n\nRECHECK: Re-read the excerpts for the exact requested facts. If any requested "
                       "fact appears, answer it using only that text. For a multi-part question, answer "
                       "the supported part even if another part is missing. Do not infer or guess.")
        payload = {
            "model": settings.ollama_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "format": OUTPUT_SCHEMA,
            "keep_alive": settings.ollama_keep_alive,
            "options": {"temperature": settings.temperature,
                        "num_ctx": settings.ollama_num_ctx,
                        "num_predict": settings.ollama_num_predict},
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/chat", data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        started = time.monotonic()
        try:
            with self.opener.open(request, timeout=settings.ollama_timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise RuntimeError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Ollama is unavailable at {self.base_url}.") from exc
        content = (result.get("message") or {}).get("content", "")
        trace("ollama_raw", model=settings.ollama_model, done_reason=result.get("done_reason"),
              response=content)
        if result.get("done_reason") == "length":
            logger.warning("Ollama output reached its token limit; validator will reject truncation.")
        logger.info("Ollama response completed in %.1fs", time.monotonic() - started)
        return content, result.get("done_reason")

    def answer(self, question: str, context: str) -> str:
        content, _ = self._generate(question, context)
        answer = validate_response(content, question, context)
        trace("answer_validation", attempt=1, accepted=answer != NOT_FOUND, answer=answer)
        parsed = parse_json(content)
        if (answer == NOT_FOUND and parsed is not None
                and _REFUSAL.fullmatch(_unescape_output(parsed["answer"]))
                and not parsed["evidence"] and self._needs_recheck(question, context)):
            trace("grounded_recheck", question=question, context=context)
            retry_content, _ = self._generate(question, context, retry_partial=True)
            retry_answer = validate_response(retry_content, question, context)
            trace("answer_validation", attempt=2, accepted=retry_answer != NOT_FOUND,
                  answer=retry_answer)
            return retry_answer
        return answer
