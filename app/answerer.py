"""Single-call Ollama answer generation with exact evidence-quote validation."""

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

logger = logging.getLogger(__name__)
NOT_FOUND = "This information is not available in the selected document."
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}, "evidence": {"type": "array", "items": {"type": "string"}}},
    "required": ["answer", "evidence"],
    "additionalProperties": False,
}
_NUMBER = re.compile(r"(?<!\w)(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?(?!\w)")
_HINDI = re.compile(r"[\u0900-\u097f]")
_REFUSAL = re.compile(re.escape(NOT_FOUND), re.IGNORECASE)
_TITLE_WORD = re.compile(r"\b[A-Z][a-z]{1,}\b|\b[A-Z]{2,}\b")
_NON_ENTITY = {"The", "This", "That", "These", "Those", "There", "It", "Its", "In", "On", "At",
               "For", "From", "By", "With", "A", "An", "And", "But", "The", "Document"}
_NON_ENTITY |= {"Technical"}
_METRICS = {"connection", "follower", "view", "like", "member", "user", "employee", "revenue",
            "fee", "price", "cost", "salary", "score", "point", "download", "subscriber", "visit",
            "impression", "click", "percent", "percentage", "year", "month", "day", "week", "hour"}
_ANSWER_GLUE = {"the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be", "been",
                "being", "has", "have", "had", "do", "does", "did", "of", "on", "in", "at", "to",
                "for", "from", "with", "by", "as", "it", "this", "that", "these", "those", "there",
                "not", "only", "part", "also", "its", "their", "his", "her", "your", "my", "we", "they",
                "you", "he", "she", "what", "which", "who", "when", "where", "how", "many", "much",
                "like",
                "skill", "technical",
                "profile", "document", "name", "called", "lists", "list", "listed", "shows", "show", "states",
                "state", "reports", "reported", "provides", "provided", "indicates", "indicated", "says", "said",
                "mentions", "mentioned", "specify", "specified", "specifies", "does", "missing", "available",
                "information", "selected", "page", "one", "there", "on", "the", "based", "begins", "began", "starts", "started",
                "count", "date", "creation", "renewal", "exist", "exists", "existence"}
# These words express ordinary grammatical paraphrases of job/date evidence. Material
# claims (names, numbers, metrics, and descriptive qualifiers) still need source support.
_SAFE_PARAPHRASE = {"currently", "current", "work", "works", "working", "employ", "employed",
                    "employment", "position", "role", "start", "starting", "since", "present",
                    "including", "include", "such", "expertise", "technology", "technologie",
                    "programming", "development", "develop", "using", "use"}
_HINGLISH_MARKERS = {"hai", "hain", "tha", "thi", "hoga", "hogi", "honge", "nahi", "nahin",
                     "kya", "kab", "kaise", "kitna", "kitni", "kitne", "kyunki", "mein", "se", "ko",
                     "par", "aur", "bhi", "wala", "wali", "wale", "mila", "milta", "milti"}


def _normalize_for_match(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _unescape_output(text: str) -> str:
    # JSON decodes ordinary escapes. Some local models double-escape line breaks; repair those.
    return text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", " ").strip()


def _english_tokens(text: str) -> set[str]:
    found = set(re.findall(r"[a-z]+", unicodedata.normalize("NFKC", text).casefold()))
    stems = set()
    for word in found:
        if word in {"january", "february", "march", "april", "june", "july", "august",
                    "september", "october", "november", "december"}:
            stems.add(word[:3])
        elif word.endswith("ies") and len(word) > 5:
            stems.add(word[:-3] + "y")
        elif word in {"working", "worked", "works"}:
            stems.add("work")
        elif word in {"scored", "scores", "scoring"}:
            stems.add("score")
        elif word in {"started", "starts", "starting"}:
            stems.add("start")
        elif word in {"began", "begins", "beginning"}:
            stems.add("begin")
        elif len(word) > 4 and word.endswith("s") and not word.endswith(("ss", "us", "is", "as")):
            stems.add(word[:-1])
        else:
            stems.add(word)
    return stems


def _capitalized_entities(text: str) -> set[str]:
    return _english_tokens(" ".join(word for word in _TITLE_WORD.findall(text)
                                    if word not in _NON_ENTITY))


def _capitalized_name_phrases(text: str) -> set[str]:
    return {" ".join(match).casefold() for match in re.findall(
        r"\b([A-Z][a-z]{2,})\s+([A-Z][a-z]{2,})\b", text)}


def _metrics_in(text: str) -> set[str]:
    found = _english_tokens(text) & _METRICS
    # "like" is also a frequent preposition ("tools like Git"). Count it as a metric
    # only in count/label constructions, where it clearly means a social-media reaction.
    if "like" in found and not re.search(
            r"\b(?:likes?\s*(?:count|number|:|=)|number\s+of\s+likes?|\d[\d,]*\s+likes?)\b",
            text, re.IGNORECASE):
        found.remove("like")
    return found


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


def validate_response(raw: str, question: str, context: str) -> str:
    result = parse_json(raw)
    if result is None:
        logger.warning("Ollama response was malformed or truncated; returning fixed abstention.")
        return NOT_FOUND

    answer = _unescape_output(result["answer"])
    evidence = [_unescape_output(value) for value in result["evidence"] if value.strip()]
    if _REFUSAL.fullmatch(answer):
        return NOT_FOUND
    # Remove the exact contradictory refusal before validating the supported answer itself.
    answer = _REFUSAL.sub("", answer).strip(" \t\r\n-–—;,")
    if not answer or not evidence:
        return NOT_FOUND

    normalized_context = _normalize_for_match(context)
    valid_quotes = [quote for quote in evidence
                    if len(quote) <= 1200 and _normalize_for_match(quote) in normalized_context]
    if not valid_quotes:
        logger.warning("Ollama answer had no evidence quote matching retrieved text.")
        return NOT_FOUND

    # A document quote is necessary but not sufficient: ensure every number asserted is in the
    # supporting quote(s). This prevents swapping a nearby follower count for a connection count.
    evidence_text = "\n".join(valid_quotes)
    evidence_numbers = {number.replace(",", "") for number in _NUMBER.findall(evidence_text)}
    answer_numbers = {number.replace(",", "") for number in _NUMBER.findall(answer)}
    if not answer_numbers.issubset(evidence_numbers):
        logger.warning("Ollama answer introduced a number absent from its evidence quotes.")
        return NOT_FOUND

    evidence_tokens = _english_tokens(evidence_text)
    answer_tokens = _english_tokens(answer)
    question_tokens = _english_tokens(question)
    requested_metrics = question_tokens & _METRICS
    if not requested_metrics.issubset(evidence_tokens):
        logger.warning("Retrieved evidence does not mention the metric asked about.")
        return NOT_FOUND
    answer_metrics = _metrics_in(answer)
    evidence_metrics = _metrics_in(evidence_text)
    if not answer_metrics.issubset(evidence_metrics):
        logger.warning("Answer changed the metric label; answer=%s evidence=%s.",
                       sorted(answer_metrics), sorted(evidence_metrics))
        return NOT_FOUND

    # Names and technical terms can be in any of the supplied excerpts, even when the
    # model's short quote omits them. Keep them grounded in the selected context.
    answer_entities = _capitalized_entities(answer)
    context_tokens = _english_tokens(context)
    question_entities = _capitalized_entities(question)
    if not (answer_entities - question_entities).issubset(context_tokens):
        logger.warning("Answer introduced names absent from evidence quotes: %s.",
                       sorted((answer_entities - question_entities) - context_tokens))
        return NOT_FOUND

    # Keep a person's name and a requested number together in one exact quote. This
    # stops a score/count for one person being assigned to a similar nearby person.
    question_people = _capitalized_name_phrases(question)
    answer_people = _capitalized_name_phrases(answer) - question_people
    if question_people and answer_people and answer_numbers:
        associated = any(
            all(_normalize_for_match(person) in _normalize_for_match(quote) for person in answer_people)
            and answer_numbers.issubset({number.replace(",", "") for number in _NUMBER.findall(quote)})
            for quote in valid_quotes
        )
        if not associated:
            logger.warning("Answer combined a different person's name with the requested number.")
            return NOT_FOUND

    # The requested output language is English. If the model ignores this, fail closed instead of
    # returning non-English prose that the frontend cannot distinguish from a grounded answer.
    if _HINDI.search(answer) or _english_tokens(answer) & _HINGLISH_MARKERS:
        logger.warning("Ollama ignored the English-only answer requirement.")
        # The evidence field is an exact quote selected by the model and already checked
        # against the selected document. Returning a short English source quote is safer
        # than refusing a fact solely because the model answered in Hindi.
        if valid_quotes:
            quote = min(valid_quotes, key=len)
            if len(quote) <= 240 and not _HINDI.search(quote):
                return quote
        return NOT_FOUND

    unsupported_terms = (_english_tokens(answer) - context_tokens - _ANSWER_GLUE
                         - _SAFE_PARAPHRASE - question_entities)
    if unsupported_terms:
        logger.warning("Answer contains terms not supported by the quoted evidence: %s",
                       ", ".join(sorted(unsupported_terms)))
        return NOT_FOUND

    # Normalize only the exact refusal already removed above and nearby separators.
    answer = re.sub(r"[ \t]{2,}", " ", answer)
    answer = re.sub(r"\n{3,}", "\n\n", answer).strip(" \t\r\n-–—;,")
    return answer or NOT_FOUND


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
        conjunction = re.search(r"\b(and|or|also|aur|bhi)\b|और|तथा", question, re.IGNORECASE)
        question_terms = _english_tokens(question) - _ANSWER_GLUE - _HINGLISH_MARKERS
        overlap = question_terms & _english_tokens(context)
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
