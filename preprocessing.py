import re
from typing import Dict, List


FILLER_WORDS = {
    "please",
    "pls",
    "kindly",
    "omg",
    "wow",
    "lol",
    "uh",
    "umm",
    "btw",
    "share",
    "forward",
    "viral",
    "breaking",
    "exclusive",
    "shocking",
    "mustwatch",
    "mustread",
    "yaar",
    "bhai",
    "arre",
    "dekho",
    "sunlo",
    "jaldi",
}

CLICKBAIT_PATTERNS = [
    r"you won't believe",
    r"must watch",
    r"watch till end",
    r"share this now",
    r"forward this",
    r"breaking news+",
    r"urgent+",
    r"shocking+",
]

URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
MENTION_RE = re.compile(r"[@#]\w+")
EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "]+",
    flags=re.UNICODE,
)


def _normalize_repeated_characters(text: str) -> str:
    # Keep expressive text but collapse extreme repetition: "soooo" -> "soo".
    return re.sub(r"(.)\1{2,}", r"\1\1", text)


def _remove_clickbait_phrases(text: str) -> str:
    result = text
    for pattern in CLICKBAIT_PATTERNS:
        result = re.sub(pattern, " ", result, flags=re.IGNORECASE)
    return result


def _remove_filler_words(tokens: List[str]) -> List[str]:
    return [token for token in tokens if token not in FILLER_WORDS]


def _dedupe_repeated_words(tokens: List[str]) -> List[str]:
    if not tokens:
        return tokens
    deduped = [tokens[0]]
    for token in tokens[1:]:
        if token != deduped[-1]:
            deduped.append(token)
    return deduped


def clean_text(text: str) -> str:
    lowered = text.lower()
    lowered = URL_RE.sub(" ", lowered)
    lowered = MENTION_RE.sub(" ", lowered)
    lowered = EMOJI_RE.sub(" ", lowered)
    lowered = _remove_clickbait_phrases(lowered)
    lowered = _normalize_repeated_characters(lowered)
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()

    tokens = lowered.split()
    tokens = _remove_filler_words(tokens)
    tokens = _dedupe_repeated_words(tokens)
    return " ".join(tokens)


def tokenize(text: str) -> List[str]:
    return [token for token in clean_text(text).split() if token]


def reduction_stats(original_text: str, cleaned_text: str) -> Dict[str, float]:
    original_chars = len(original_text)
    cleaned_chars = len(cleaned_text)
    original_tokens = max(1, len(original_text.split()))
    cleaned_tokens = len(cleaned_text.split())

    char_reduction = ((original_chars - cleaned_chars) / max(1, original_chars)) * 100.0
    token_reduction = ((original_tokens - cleaned_tokens) / original_tokens) * 100.0
    return {
        "original_chars": float(original_chars),
        "cleaned_chars": float(cleaned_chars),
        "original_tokens": float(original_tokens),
        "cleaned_tokens": float(cleaned_tokens),
        "char_reduction_pct": round(char_reduction, 2),
        "token_reduction_pct": round(token_reduction, 2),
    }
