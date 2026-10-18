"""
plagiarism.py
-------------
Plagiarism / similarity detection module.

Uses TF-IDF vectorization and cosine similarity to compare documents
and identify similar sentence pairs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing import preprocess_text, get_sentences


# ---------------------------------------------------------------------------
# Similarity thresholds
# ---------------------------------------------------------------------------
SIMILARITY_THRESHOLDS = {
    "Low": (0.0, 0.30),
    "Moderate": (0.30, 0.60),
    "High": (0.60, 0.80),
    "Very High": (0.80, 1.01),
}


def get_similarity_label(score: float) -> str:
    """Return a human-readable similarity label for a given score (0–1)."""
    for label, (low, high) in SIMILARITY_THRESHOLDS.items():
        if low <= score < high:
            return label
    return "Unknown"


# ---------------------------------------------------------------------------
# Document-level similarity
# ---------------------------------------------------------------------------

def compute_document_similarity(
    documents: list[str],
    doc_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute pairwise cosine similarity between all documents.

    Parameters
    ----------
    documents : list[str]
        Raw text for each document.
    doc_names : list[str]
        Corresponding filenames / labels.

    Returns
    -------
    similarity_matrix : pd.DataFrame
        Square DataFrame of similarity scores (0–100 %).
    pairwise_df : pd.DataFrame
        Long-form table with columns:
        [Document A, Document B, Similarity (%), Status]
    """
    if len(documents) < 2:
        raise ValueError("At least two documents are required for similarity analysis.")

    # Preprocess each document
    preprocessed = [preprocess_text(doc) for doc in documents]

    # TF-IDF vectorisation
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    tfidf_matrix = vectorizer.fit_transform(preprocessed)

    # Cosine similarity matrix
    sim_matrix = cosine_similarity(tfidf_matrix)  # shape: (n, n)

    # Build a labelled square DataFrame (percentages)
    sim_pct = np.round(sim_matrix * 100, 2)
    matrix_df = pd.DataFrame(sim_pct, index=doc_names, columns=doc_names)

    # Build pairwise long-form table (upper triangle only, no self-comparison)
    n = len(doc_names)
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            score = sim_pct[i, j]
            rows.append(
                {
                    "Document A": doc_names[i],
                    "Document B": doc_names[j],
                    "Similarity (%)": score,
                    "Status": get_similarity_label(score / 100),
                }
            )

    pairwise_df = pd.DataFrame(rows)
    return matrix_df, pairwise_df


# ---------------------------------------------------------------------------
# Sentence-level similarity
# ---------------------------------------------------------------------------

def compute_sentence_similarity(
    doc_a_text: str,
    doc_b_text: str,
    doc_a_name: str,
    doc_b_name: str,
    threshold: float = 0.70,
) -> list[dict]:
    """
    Find sentence pairs between two documents that exceed the similarity threshold.

    Parameters
    ----------
    doc_a_text : str
        Raw text of the first document.
    doc_b_text : str
        Raw text of the second document.
    doc_a_name : str
        Label / filename of the first document.
    doc_b_name : str
        Label / filename of the second document.
    threshold : float
        Cosine similarity threshold (0–1) above which a match is reported.

    Returns
    -------
    list[dict]
        Each dict contains:
        {doc_a, sentence_a, doc_b, sentence_b, similarity}
    """
    sentences_a = get_sentences(doc_a_text)
    sentences_b = get_sentences(doc_b_text)

    if not sentences_a or not sentences_b:
        return []

    # Preprocess sentences for TF-IDF
    preprocessed_a = [preprocess_text(s) for s in sentences_a]
    preprocessed_b = [preprocess_text(s) for s in sentences_b]

    # Build a combined corpus to fit the vectorizer
    all_sentences = preprocessed_a + preprocessed_b
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    vectorizer.fit(all_sentences)

    vec_a = vectorizer.transform(preprocessed_a)
    vec_b = vectorizer.transform(preprocessed_b)

    # Cosine similarity: shape (len_a, len_b)
    sim_matrix = cosine_similarity(vec_a, vec_b)

    matches = []
    for i, row in enumerate(sim_matrix):
        for j, score in enumerate(row):
            if score >= threshold:
                matches.append(
                    {
                        "doc_a": doc_a_name,
                        "sentence_a": sentences_a[i],
                        "doc_b": doc_b_name,
                        "sentence_b": sentences_b[j],
                        "similarity": round(score * 100, 2),
                    }
                )

    # Sort by similarity descending
    matches.sort(key=lambda x: x["similarity"], reverse=True)

    # Remove duplicates (same sentence pair from different index combos)
    seen: set[tuple[str, str]] = set()
    unique_matches = []
    for m in matches:
        key = (m["sentence_a"], m["sentence_b"])
        if key not in seen:
            seen.add(key)
            unique_matches.append(m)

    return unique_matches


def find_all_sentence_matches(
    documents: list[str],
    doc_names: list[str],
    threshold: float = 0.70,
) -> list[dict]:
    """
    Find similar sentence pairs across ALL document pairs.

    Parameters
    ----------
    documents : list[str]
        Raw document texts.
    doc_names : list[str]
        Document labels.
    threshold : float
        Similarity threshold (0–1).

    Returns
    -------
    list[dict]
        Combined list of matching sentence dicts.
    """
    n = len(documents)
    all_matches: list[dict] = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs = compute_sentence_similarity(
                documents[i],
                documents[j],
                doc_names[i],
                doc_names[j],
                threshold=threshold,
            )
            all_matches.extend(pairs)
    return all_matches

