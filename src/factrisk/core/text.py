from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from itertools import combinations
from typing import Any


_SPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_NUMBER_RE = re.compile(r"[+-]?\d+(?:[.,]\d+)?")
_WORD_RE = re.compile(r"\b[^\W\d_]+(?:-[^\W\d_]+)?\b", flags=re.IGNORECASE)

NEGATIVE_MARKERS = {
    "not", "no", "never", "cannot", "can't", "dont", "don't", "doesnt",
    "doesn't", "didnt", "didn't", "wont", "won't", "isnt", "isn't",
    "nothing", "nobody", "no one", "neither", "without", "hardly",
}

NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90", "first": "1", "second": "2",
    "third": "3", "fourth": "4", "fifth": "5", "sixth": "6",
    "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10",
    "once": "1", "twice": "2", "double": "2", "triple": "3",
    "single": "1", "both": "2", "dozen": "12",
}


def normalize_text(text: str | None, *, keep_punct: bool = False) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).lower().strip()
    if not keep_punct:
        value = _PUNCT_RE.sub(" ", value)
    return _SPACE_RE.sub(" ", value).strip()


def text_units(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    tokens = normalized.split()
    return tokens if len(tokens) > 1 else list(normalized)


def edit_distance(left: Sequence[str], right: Sequence[str]) -> int:
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for index, left_item in enumerate(left, start=1):
        current = [index]
        for other_index, right_item in enumerate(right, start=1):
            cost = int(left_item != right_item)
            current.append(
                min(
                    previous[other_index] + 1,
                    current[other_index - 1] + 1,
                    previous[other_index - 1] + cost,
                )
            )
        previous = current
    return previous[-1]


def normalized_distance(left: str, right: str) -> float:
    a, b = text_units(left), text_units(right)
    return edit_distance(a, b) / max(len(a), len(b), 1)


def pairwise_mean_distance(texts: list[str]) -> float:
    present = [text for text in texts if normalize_text(text)]
    if len(present) < 2:
        return float("nan")
    distances = [normalized_distance(a, b) for a, b in combinations(present, 2)]
    return sum(distances) / len(distances)


def extract_numbers(text: str) -> set[str]:
    # Return whole mentions: 'twenty one' must not match the fact 'twenty'.
    from factrisk.core.numbers import number_mentions

    return {item["value"] for item in number_mentions(text)}


def text_is_negative(text: str) -> bool:
    normalized = normalize_text(text, keep_punct=True)
    return any(
        re.search(rf"(?<!\w){re.escape(normalize_text(marker, keep_punct=True))}(?!\w)", normalized)
        for marker in NEGATIVE_MARKERS
    )


def slot_polarity(slot: Any) -> bool | None:
    surfaces = slot_surfaces(slot)
    if not surfaces:
        return None
    joined = " ".join(surfaces)
    if text_is_negative(joined):
        return True
    affirmative = {"affirmative", "positive", "yes", "will", "present", "again"}
    normalized = set(normalize_text(joined).split())
    if normalized & affirmative:
        return False
    return None


def slot_surfaces(slot: Any) -> list[str]:
    if slot is None:
        return []
    if isinstance(slot, (str, int, float)):
        return [str(slot)] if str(slot) else []
    if isinstance(slot, (list, tuple, set)):
        values: list[str] = []
        for item in slot:
            values.extend(slot_surfaces(item))
        return _dedupe(values)
    if isinstance(slot, dict):
        values = []
        for key in ("value", "surface", "text", "translation", "aliases"):
            if key in slot:
                values.extend(slot_surfaces(slot[key]))
        return _dedupe(values)
    return [str(slot)]


def slot_correct(text: str, expected_slot: Any, fact_type: str) -> bool:
    fact_type = fact_type.lower()
    surfaces = slot_surfaces(expected_slot)
    if not surfaces:
        return False
    if fact_type == "number":
        observed = extract_numbers(text)
        expected: set[str] = set()
        for surface in surfaces:
            expected |= extract_numbers(surface)
        if observed and expected:
            return bool(observed & expected)
    if fact_type == "negation":
        polarity = slot_polarity(surfaces)
        if polarity is not None:
            return text_is_negative(text) == polarity
    normalized = normalize_text(text)
    return any(
        bool(re.search(rf"(?<!\w){re.escape(normalize_text(surface))}(?!\w)", normalized))
        for surface in surfaces
        if normalize_text(surface)
    )


def unsupported_contrast(
    text: str,
    expected_slot: Any,
    contrast_slot: Any,
    fact_type: str,
) -> bool:
    fact_type = fact_type.lower()
    if fact_type == "number":
        observed = extract_numbers(text)
        expected = _slot_numbers(expected_slot)
        contrast = _slot_numbers(contrast_slot)
        return bool(observed & contrast) and not bool(observed & expected)
    if fact_type == "negation":
        expected_polarity = slot_polarity(expected_slot)
        contrast_polarity = slot_polarity(contrast_slot)
        if expected_polarity is not None:
            return text_is_negative(text) != expected_polarity
        if contrast_polarity is not None:
            return text_is_negative(text) == contrast_polarity
    return slot_correct(text, contrast_slot, fact_type) and not slot_correct(
        text, expected_slot, fact_type
    )


def fact_signature(text: str, fact_type: str) -> tuple[str, ...]:
    if fact_type.lower() == "number":
        return tuple(sorted(extract_numbers(text)))
    if fact_type.lower() == "negation":
        return ("negative" if text_is_negative(text) else "affirmative",)
    return tuple()


def token_coverage(reference_like: str, hypothesis: str) -> tuple[float, float]:
    source = set(text_units(reference_like))
    target = set(text_units(hypothesis))
    if not source and not target:
        return 1.0, 1.0
    overlap = len(source & target)
    precision = overlap / max(len(target), 1)
    recall = overlap / max(len(source), 1)
    return precision, recall


def _slot_numbers(slot: Any) -> set[str]:
    values: set[str] = set()
    for surface in slot_surfaces(slot):
        values |= extract_numbers(surface)
    return values


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))
