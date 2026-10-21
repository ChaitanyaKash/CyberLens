"""Conservative, language-independent cleaning. Rules are NOT complete anonymization."""
import hashlib
import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    kind: str
    priority: int


# Higher-specificity spans win overlaps. Actual values never enter an audit log.
RULES = [
    ("URL", 100, r"(?:https?://|www\.)[^\s<>]+"),
    ("EMAIL", 95, r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+(?!\w)"),
    ("UPI_ID", 90, r"(?<![\w.@])[a-z0-9._-]{2,}@[a-z][a-z0-9]{1,}(?![\w.])"),
    ("PAN", 85, r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),
    ("AADHAAR_CANDIDATE", 80, r"(?<!\d)[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}(?!\d)"),
    ("PHONE", 75, r"(?<!\w)(?:\+?91[ -]?)?[6-9]\d{4}[ -]?\d{5}(?!\w)"),
]
CONTEXT_RULES = [
    ("OTP", r"\b(?:otp|one[ -]?time password|verification code)\s*(?:is|hai|tha|:|=|-)?\s*(\d{4,8})\b"),
    ("PIN", r"\b(?:pin|mpin|cvv)\s*(?:is|hai|tha|:|=|-)?\s*(\d{3,6})\b"),
    ("ACCOUNT", r"\b(?:account|a/c|acc|khata)(?:\s*(?:number|no\.?|id))?\s*(?:is|hai|:|=|-)?\s*(\d[\d -]{6,22}\d)\b"),
    ("CARD", r"\b(?:card)(?:\s*(?:number|no\.?))?\s*(?:is|hai|:|=|-)?\s*(\d[\d -]{11,22}\d)\b"),
]
SAFE_VARIANTS = {"mujhy": "mujhe", "mujhey": "mujhe", "plz": "please", "pls": "please"}
TOKEN_RE = re.compile(r"<[A-Z_]+>|[^\W_]+(?:['’-][^\W_]+)*", re.UNICODE | re.IGNORECASE)


def mask_sensitive(text, extra_spans=()):
    """Merge optional NER spans (start, end, kind), then replace right to left.

    Does not claim names/addresses are removed without an external NER detector.
    Returns masked text and counts, never a raw-value map.
    """
    text = str(text)
    candidates = []
    for kind, priority, pattern in RULES:
        candidates.extend(Span(m.start(), m.end(), kind, priority)
                          for m in re.finditer(pattern, text, re.I))
    for kind, pattern in CONTEXT_RULES:
        candidates.extend(Span(m.start(1), m.end(1), kind, 110)
                          for m in re.finditer(pattern, text, re.I))
    for start, end, kind in extra_spans:
        if not (0 <= start < end <= len(text)) or not re.fullmatch(r"[A-Z_]+", kind):
            raise ValueError("Invalid NER span")
        candidates.append(Span(start, end, kind, 70))
    selected = []
    for span in sorted(candidates, key=lambda s: (-s.priority, -(s.end-s.start), s.start)):
        if not any(span.start < s.end and s.start < span.end for s in selected):
            selected.append(span)
    counts = {}
    for span in sorted(selected, key=lambda s: s.start, reverse=True):
        counts[span.kind] = counts.get(span.kind, 0) + 1
        text = text[:span.start] + f"<{span.kind}>" + text[span.end:]
    return text, counts


def normalize(text, variants=False):
    text = unicodedata.normalize("NFKC", str(text))
    text = re.sub(r"[\u200b\ufeff]", "", text)
    text, _ = mask_sensitive(text)
    text = re.sub(r"\s+", " ", text).strip()
    if variants:
        text = re.sub(r"\b\w+\b", lambda m: SAFE_VARIANTS.get(m[0].lower(), m[0]), text)
    return text


def tokens(text):
    return [m.group().lower() for m in TOKEN_RE.finditer(text)]


def fingerprint(text, already_normalized=False):
    # Numeric differences and punctuation must not split otherwise identical templates.
    text = re.sub(r"\d+", " NUM ", (text if already_normalized else normalize(text)).lower())
    text = " ".join(tokens(text))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def corrupt(text, seed=42, rate=0.12):
    """Synthetic adjacent-letter swaps; never corrupt placeholders or numbers."""
    import random
    rng = random.Random(seed)
    def edit(m):
        word = m[0]
        if word.startswith("<") or len(word) < 5 or not word.isalpha() or rng.random() >= rate:
            return word
        i = rng.randrange(1, len(word)-2)
        return word[:i] + word[i+1] + word[i] + word[i+2:]
    return TOKEN_RE.sub(edit, text)
