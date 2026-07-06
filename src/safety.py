"""
Safety Layer: Confidence-Gated Confirm-Back

The problem: errors in the ASR + MT cascade compound invisibly.
The user is non-literate — they can't read a transcript to catch mistakes.

The solution: back-translate what we understood (zu → en → zu) and
compare against what the user originally said. If the round-trip
diverges too much, we ask them to confirm before proceeding to the LLM.

Three confidence tiers:
  - HIGH  (>= high_threshold):  proceed silently
  - MED   (>= low_threshold):   implicit confirm — echo back while proceeding
  - LOW   (< low_threshold):    explicit confirm-back — ask before proceeding

This is where the real novelty lives. Nobody has benchmarked confirm-back
for non-literate voice users.
"""

import re
from difflib import SequenceMatcher
from enum import Enum
from dataclasses import dataclass
from rich.console import Console

from src.config import settings

console = Console()


class Confidence(Enum):
    HIGH = "high"
    MED = "med"
    LOW = "low"


@dataclass
class SafetyResult:
    confidence: Confidence
    similarity: float
    original: str
    back_translated: str


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace for fair comparison."""
    text = text.lower().strip()
    text = re.sub(r"[.,!?;:'\"()\[\]]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def similarity(a: str, b: str) -> float:
    """
    Return a similarity score between 0.0 and 1.0.
    Uses sequence matching on normalized strings.
    """
    a_norm = _normalize(a)
    b_norm = _normalize(b)

    if not a_norm or not b_norm:
        return 0.0

    return SequenceMatcher(None, a_norm, b_norm).ratio()


def check_confidence(
    original: str,
    back_translated: str,
    high_threshold: float = None,
    low_threshold: float = None,
) -> SafetyResult:
    """
    Compare original Zulu against back-translated Zulu.
    Returns a confidence tier and similarity score.
    """
    high = high_threshold or settings.safety_high_threshold
    low = low_threshold or settings.safety_low_threshold

    sim = similarity(original, back_translated)

    if sim >= high:
        conf = Confidence.HIGH
    elif sim >= low:
        conf = Confidence.MED
    else:
        conf = Confidence.LOW

    return SafetyResult(
        confidence=conf,
        similarity=sim,
        original=original,
        back_translated=back_translated,
    )


def format_confidence(result: SafetyResult) -> str:
    """Format the safety result for display."""
    color = {
        Confidence.HIGH: "green",
        Confidence.MED: "yellow",
        Confidence.LOW: "red",
    }[result.confidence]

    return (
        f"[{color}]Confidence: {result.confidence.value.upper()} "
        f"({result.similarity:.2f})[/{color}]"
    )


def is_affirmative(text: str) -> bool:
    """
    Check if a user's Zulu response means 'yes'.
    Handles common Zulu affirmative words and their variants.
    """
    text = _normalize(text)
    affirmatives = [
        "yebo",       # yes
        "ye",
        "ehe",        # yes/agreement
        "kunjalo",    # that's right
        "kulungile",  # it's okay/correct
        "yes",        # code-switching
        "ok",
        "okay",
    ]
    return any(word in text.split() for word in affirmatives)


def is_negative(text: str) -> bool:
    """Check if a user's Zulu response means 'no'."""
    text = _normalize(text)
    negatives = [
        "cha",         # no
        "qha",
        "hayi",        # no (informal)
        "akulona",     # it's not
        "no",          # code-switching
    ]
    return any(word in text.split() for word in negatives)


# Standalone test
if __name__ == "__main__":
    console.print("[bold]Safety Layer Test[/bold]\n")

    test_cases = [
        ("ngiyabonga", "ngiyabonga", "identical"),
        ("ngiyabonga", "ngiyabonga kakhulu", "close"),
        ("bengifuna ukuvula i-bank account", "ngifuna ukuvula ibhange", "similar meaning"),
        ("bengifuna ukubuza umbuzo", "bathanda ukudlala ibhola", "totally different"),
    ]

    for orig, back, desc in test_cases:
        result = check_confidence(orig, back)
        console.print(f"[dim]{desc}[/dim]")
        console.print(f"  Original:        '{orig}'")
        console.print(f"  Back-translated: '{back}'")
        console.print(f"  {format_confidence(result)}\n")
