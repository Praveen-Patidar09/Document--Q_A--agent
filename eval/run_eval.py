"""Repeatable retrieval/answer validation evaluation; --live opts into actual Ollama calls."""

import argparse
import json
import logging
import re
from pathlib import Path

from app.answerer import NOT_FOUND, OllamaAnswerer, validate_response
from app.models import Chunk
from app.retrieval import prepare_context, rank_candidates
from app.config import settings

_EVAL_GLUE = {"the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be",
              "been", "being", "has", "have", "had", "do", "does", "did", "of", "on",
              "in", "at", "to", "for", "from", "with", "by", "as", "it", "this", "that",
              "these", "those", "there", "not", "only", "part", "also", "its", "their", "his",
              "her", "your", "my", "we", "they", "you", "he", "she", "what", "which", "who",
              "when", "where", "how", "many", "much", "profile", "document", "name", "called",
              "lists", "list", "listed", "shows", "show", "states", "state", "reports", "reported",
              "provides", "provided", "indicates", "indicated", "says", "said", "mentions", "mentioned",
              "information", "page", "one", "based", "count", "starts", "start"}


def _core_terms(text: str) -> set[str]:
    result = set()
    for word in re.findall(r"[A-Za-z]+|\d+(?:,\d{3})*(?:\.\d+)?", text):
        word = word.casefold()
        if word in _EVAL_GLUE:
            continue
        if word.isalpha() and len(word) > 4 and word.endswith("s") and not word.endswith(("ss", "us", "is", "as")):
            word = word[:-1]
        result.add(word)
    return result


def _matches_expected(actual: str, case: dict) -> bool:
    if not case["answerable"]:
        return actual == NOT_FOUND
    if actual == NOT_FOUND:
        return False
    required = set(case.get("expected_core", [])) or _core_terms(case["expected_answer"])
    return required.issubset(_core_terms(actual))


def run(live: bool = False) -> dict:
    cases = json.loads((Path(__file__).parent / "cases.json").read_text(encoding="utf-8"))
    counts = {"retrieval_failures": 0, "incorrect_refusals": 0, "unsupported_answers": 0,
             "answer_accuracy": 0, "cases": len(cases)}
    details = []
    answerer = OllamaAnswerer() if live else None

    for case in cases:
        rows = [(Chunk(text, {"filename": case.get("filename", f"{case['id']}.txt"),
                              "page": case.get("page"), "chunk": index}), score)
                for index, (text, score) in enumerate(case["passages"])]
        ranked = rank_candidates(case["question"], rows, limit=len(rows), min_similarity=0.0)
        selected_texts = [item.chunk.text for item in ranked]
        if case["passages"][case["expected_chunk"]][0] not in selected_texts:
            counts["retrieval_failures"] += 1
        context = prepare_context(ranked, case["question"])
        if live:
            try:
                actual = answerer.answer(case["question"], context)
            except Exception as exc:
                return {"mode": "live_ollama", "status": "unavailable",
                        "error_type": type(exc).__name__, "error": str(exc),
                        "cases_run": len(details), "cases_total": len(cases),
                        "metrics": None, "details": details}
        else:
            raw = json.dumps(case["response"], ensure_ascii=False)
            actual = validate_response(raw, case["question"], context)
        expected = case["expected_answer"]
        passed = _matches_expected(actual, case)
        if passed:
            counts["answer_accuracy"] += 1
        if case["answerable"] and actual == NOT_FOUND:
            counts["incorrect_refusals"] += 1
        if not case["answerable"] and actual != NOT_FOUND:
            counts["unsupported_answers"] += 1
        if case["answerable"] and actual != NOT_FOUND and not passed:
            counts.setdefault("answer_mismatches", 0)
            counts["answer_mismatches"] += 1
        details.append({"id": case["id"], "expected": expected, "actual": actual,
                        "passed": passed, "classification":
                        "refusal" if actual == NOT_FOUND else ("supported" if passed else "unsupported_or_incomplete")})

    counts["answer_accuracy_percent"] = round(100 * counts["answer_accuracy"] / max(1, len(cases)), 1)
    return {"mode": "live_ollama" if live else "deterministic_fixture",
            "metrics": counts, "details": details}


if __name__ == "__main__":
    if settings.diagnostics:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Send eval prompts to configured Ollama model.")
    args = parser.parse_args()
    print(json.dumps(run(args.live), ensure_ascii=False, indent=2))
