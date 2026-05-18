"""Text preprocessing helpers for Plaso/log2timeline timeline messages."""

from __future__ import annotations

import re
from collections.abc import Iterable

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


CUSTOM_STOP_WORDS = {
    "access",
    "application",
    "data",
    "default",
    "entry",
    "file",
    "files",
    "flags",
    "folder",
    "local",
    "microsoft",
    "program",
    "system",
    "time",
    "type",
    "user",
    "users",
    "windows",
}

STOP_WORDS = frozenset(ENGLISH_STOP_WORDS.union(CUSTOM_STOP_WORDS))

URL_RE = re.compile(r"\b(?:https?|ftp)://[^\s,;|<>\"]+", re.IGNORECASE)
PATH_RE = re.compile(
    r"""(?ix)
    \b
    (?:
        [a-z]:
        | ntfs:
        | gzip:
        | sysvol:
        | recycle_bin:
    )?
    [\\/]
    [^\s,;|<>"']+
    """
)
TIMESTAMP_RE = re.compile(
    r"\b\d{4}[-/:]\d{1,2}[-/:]\d{1,2}(?:[T\s]\d{1,2}:\d{2}:\d{2}(?:\.\d+)?)?\b",
    re.IGNORECASE,
)
HEX_RE = re.compile(r"\b(?:0x)?[a-f0-9]{9,}\b", re.IGNORECASE)
NUMBER_RE = re.compile(r"\b\d+\b")
TOKEN_RE = re.compile(r"[a-z_][a-z_]+")
SOURCE_RE = re.compile(r"[^a-z0-9]+")


def clean_message(
    message: object,
    *,
    max_chars: int = 2000,
    min_token_length: int = 2,
    stop_words: Iterable[str] = STOP_WORDS,
) -> str:
    """Normalize a raw timeline message into a compact token string."""
    text = "" if message is None else str(message)
    text = text[:max_chars].lower()
    text = URL_RE.sub(" url_token ", text)
    text = PATH_RE.sub(" path_token ", text)
    text = TIMESTAMP_RE.sub(" time_token ", text)
    text = HEX_RE.sub(" hex_token ", text)
    text = NUMBER_RE.sub(" num_token ", text)

    stop_word_set = set(stop_words)
    tokens = [
        token
        for token in TOKEN_RE.findall(text)
        if len(token) >= min_token_length and token not in stop_word_set
    ]
    return " ".join(tokens)


def normalize_source(source: object) -> str:
    """Convert a Plaso source value into a stable source token."""
    value = "" if source is None else str(source).strip().lower()
    value = SOURCE_RE.sub("_", value).strip("_")
    return f"source_{value or 'unknown'}"


def build_clustering_text(source: object, message: object) -> str:
    """Build the final text used by TF-IDF: source token plus cleaned message."""
    cleaned = clean_message(message)
    source_token = normalize_source(source)
    return f"{source_token} {cleaned}".strip()
