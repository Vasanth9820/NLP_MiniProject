"""
preprocessing.py
----------------
Text preprocessing module for the AI Content & Plagiarism Detection System.

Provides functions for:
  - Cleaning raw text
  - Tokenizing text
  - Removing stopwords
  - Full preprocessing pipeline
"""

import re
import string
import nltk


def _ensure_nltk_resources() -> None:
    """Download required NLTK data packages if they are not already present."""
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
    ]
    for path, package in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)


# Run on import so every function is ready to use immediately.
_ensure_nltk_resources()

from nltk.corpus import stopwords  # noqa: E402  (imported after download)
from nltk.tokenize import word_tokenize, sent_tokenize  # noqa: E402
from nltk.stem import WordNetLemmatizer  # noqa: E402

_STOP_WORDS: set = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Clean raw text by:
      - Converting to lowercase
      - Removing URLs
      - Removing HTML tags
      - Removing punctuation and special characters
      - Collapsing extra whitespace

    The original text is NOT modified; a new string is returned.

    Parameters
    ----------
    text : str
        Raw input text.

    Returns
    -------
    str
        Cleaned text string.
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove punctuation
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    # Remove digits (optional – keeps letters only for similarity comparison)
    text = re.sub(r"\d+", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_text(text: str) -> list[str]:
    """
    Tokenize text into individual word tokens.

    Parameters
    ----------
    text : str
        Input text (can be raw or already cleaned).

    Returns
    -------
    list[str]
        List of word tokens.
    """
    if not text:
        return []
    return word_tokenize(text)


def remove_stopwords(tokens: list[str]) -> list[str]:
    """
    Remove common English stopwords from a list of tokens.

    Parameters
    ----------
    tokens : list[str]
        Word tokens.

    Returns
    -------
    list[str]
        Tokens with stopwords removed.
    """
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    """
    Lemmatize a list of tokens to their base/dictionary form.

    Parameters
    ----------
    tokens : list[str]
        Word tokens.

    Returns
    -------
    list[str]
        Lemmatized tokens.
    """
    return [_LEMMATIZER.lemmatize(t) for t in tokens]


def preprocess_text(text: str, lemmatize: bool = True) -> str:
    """
    Full preprocessing pipeline that returns a single cleaned string
    suitable for TF-IDF vectorization or ML feature extraction.

    Pipeline:
      clean_text → tokenize → remove stopwords → (optional) lemmatize → join

    Parameters
    ----------
    text : str
        Raw input text.
    lemmatize : bool
        Whether to apply lemmatization (default: True).

    Returns
    -------
    str
        Space-joined preprocessed tokens.
    """
    cleaned = clean_text(text)
    tokens = tokenize_text(cleaned)
    tokens = remove_stopwords(tokens)
    if lemmatize:
        tokens = lemmatize_tokens(tokens)
    return " ".join(tokens)


def get_sentences(text: str) -> list[str]:
    """
    Split text into individual sentences using NLTK's sentence tokenizer.

    Parameters
    ----------
    text : str
        Raw input text.

    Returns
    -------
    list[str]
        List of sentence strings.
    """
    if not text or not text.strip():
        return []
    sentences = sent_tokenize(text)
    # Filter out very short / empty sentences
    return [s.strip() for s in sentences if len(s.strip()) > 10]

