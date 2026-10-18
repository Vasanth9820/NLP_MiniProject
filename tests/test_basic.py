"""
tests/test_basic.py
-------------------
Basic unit tests for the AI Content & Plagiarism Detection System.

Run with:
    python -m pytest tests/test_basic.py -v

or:
    python tests/test_basic.py
"""

from __future__ import annotations

import sys
import pathlib

# Ensure the project root is on the path
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import unittest

from src.preprocessing import clean_text, tokenize_text, preprocess_text, get_sentences
from src.plagiarism import compute_document_similarity, get_similarity_label
from src.document_parser import parse_txt, parse_document


class TestPreprocessing(unittest.TestCase):
    """Tests for the preprocessing module."""

    def test_clean_text_lowercase(self):
        """clean_text should convert text to lowercase."""
        result = clean_text("Hello World")
        self.assertEqual(result, "hello world")

    def test_clean_text_removes_urls(self):
        """clean_text should remove URLs."""
        result = clean_text("Visit https://example.com for more info.")
        self.assertNotIn("https", result)
        self.assertNotIn("example.com", result)

    def test_clean_text_removes_punctuation(self):
        """clean_text should remove punctuation."""
        result = clean_text("Hello, world! How are you?")
        for char in ".,!?;:":
            self.assertNotIn(char, result)

    def test_clean_text_empty_string(self):
        """clean_text should handle empty strings."""
        self.assertEqual(clean_text(""), "")

    def test_clean_text_none_like(self):
        """clean_text should handle non-string gracefully."""
        self.assertEqual(clean_text(None), "")  # type: ignore

    def test_clean_text_whitespace(self):
        """clean_text should collapse multiple spaces."""
        result = clean_text("hello    world")
        self.assertNotIn("  ", result)

    def test_tokenize_text_basic(self):
        """tokenize_text should split text into tokens."""
        tokens = tokenize_text("machine learning is great")
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)

    def test_tokenize_text_empty(self):
        """tokenize_text should return an empty list for empty input."""
        self.assertEqual(tokenize_text(""), [])

    def test_preprocess_text_returns_string(self):
        """preprocess_text should return a non-empty string for valid input."""
        result = preprocess_text("Machine learning algorithms are fascinating")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_preprocess_text_removes_stopwords(self):
        """preprocess_text output should not contain common stopwords."""
        result = preprocess_text("the quick brown fox")
        tokens = result.split()
        self.assertNotIn("the", tokens)

    def test_preprocess_empty_text(self):
        """preprocess_text should handle empty input gracefully."""
        result = preprocess_text("")
        self.assertEqual(result, "")

    def test_get_sentences(self):
        """get_sentences should split text into multiple sentences."""
        text = "Machine learning is amazing. It has changed the world. Many companies use it."
        sentences = get_sentences(text)
        self.assertIsInstance(sentences, list)
        self.assertGreaterEqual(len(sentences), 2)

    def test_get_sentences_empty(self):
        """get_sentences should return an empty list for empty input."""
        self.assertEqual(get_sentences(""), [])


class TestPlagiarism(unittest.TestCase):
    """Tests for the plagiarism detection module."""

    def setUp(self):
        self.doc1 = (
            "Machine learning is a branch of artificial intelligence. "
            "It enables computers to learn from data without being explicitly programmed."
        )
        self.doc2 = (
            "Machine learning is a field of artificial intelligence. "
            "It allows systems to learn from data without explicit programming."
        )
        self.doc3 = (
            "The history of ancient Rome is fascinating. "
            "The Roman Empire expanded across much of Europe and North Africa."
        )

    def test_compute_similarity_returns_dataframes(self):
        """compute_document_similarity should return two DataFrames."""
        import pandas as pd
        matrix_df, pairwise_df = compute_document_similarity(
            [self.doc1, self.doc2, self.doc3],
            ["Doc1", "Doc2", "Doc3"],
        )
        self.assertIsInstance(matrix_df, pd.DataFrame)
        self.assertIsInstance(pairwise_df, pd.DataFrame)

    def test_similar_docs_score_higher(self):
        """Similar documents should have a higher similarity score than dissimilar ones."""
        _, pairwise_df = compute_document_similarity(
            [self.doc1, self.doc2, self.doc3],
            ["Doc1", "Doc2", "Doc3"],
        )
        sim_1_2 = pairwise_df.loc[
            (pairwise_df["Document A"] == "Doc1") & (pairwise_df["Document B"] == "Doc2"),
            "Similarity (%)",
        ].values[0]
        sim_1_3 = pairwise_df.loc[
            (pairwise_df["Document A"] == "Doc1") & (pairwise_df["Document B"] == "Doc3"),
            "Similarity (%)",
        ].values[0]
        self.assertGreater(sim_1_2, sim_1_3)

    def test_identical_docs_max_similarity(self):
        """Identical documents should have 100% similarity."""
        _, pairwise_df = compute_document_similarity(
            [self.doc1, self.doc1],
            ["A", "B"],
        )
        score = pairwise_df["Similarity (%)"].values[0]
        self.assertAlmostEqual(score, 100.0, delta=0.1)

    def test_requires_at_least_two_docs(self):
        """compute_document_similarity should raise ValueError with < 2 docs."""
        with self.assertRaises(ValueError):
            compute_document_similarity([self.doc1], ["Doc1"])

    def test_pairwise_df_has_correct_columns(self):
        """Pairwise DataFrame should have expected columns."""
        _, pairwise_df = compute_document_similarity(
            [self.doc1, self.doc2],
            ["A", "B"],
        )
        expected = {"Document A", "Document B", "Similarity (%)", "Status"}
        self.assertTrue(expected.issubset(set(pairwise_df.columns)))

    def test_similarity_labels(self):
        """get_similarity_label should return correct labels."""
        self.assertEqual(get_similarity_label(0.10), "Low")
        self.assertEqual(get_similarity_label(0.45), "Moderate")
        self.assertEqual(get_similarity_label(0.70), "High")
        self.assertEqual(get_similarity_label(0.90), "Very High")


class TestDocumentParser(unittest.TestCase):
    """Tests for document parsing."""

    def test_parse_txt_basic(self):
        """parse_txt should decode bytes to string."""
        content = b"Hello, this is a test document."
        result = parse_txt(content)
        self.assertIn("Hello", result)

    def test_parse_txt_utf8(self):
        """parse_txt should handle UTF-8 encoding."""
        content = "Café résumé naïve".encode("utf-8")
        result = parse_txt(content)
        self.assertIsInstance(result, str)

    def test_parse_document_txt(self):
        """parse_document should dispatch to txt parser for .txt files."""
        content = b"Simple text content."
        result = parse_document("test.txt", content)
        self.assertIn("Simple text", result)

    def test_parse_document_unsupported(self):
        """parse_document should raise ValueError for unknown extensions."""
        with self.assertRaises(ValueError):
            parse_document("file.xyz", b"content")

    def test_parse_document_no_extension(self):
        """parse_document should raise ValueError for files without extension."""
        with self.assertRaises(ValueError):
            parse_document("filename", b"content")


class TestAIDetector(unittest.TestCase):
    """Tests for the AI content detection module."""

    def test_detector_loads(self):
        """AIDetector should initialise without raising exceptions."""
        from src.ai_detector import AIDetector
        detector = AIDetector()
        self.assertIsInstance(detector.is_ready, bool)

    def test_model_status_message(self):
        """get_model_status_message should return a non-empty string."""
        from src.ai_detector import get_detector
        detector = get_detector()
        msg = detector.get_model_status_message()
        self.assertIsInstance(msg, str)
        self.assertGreater(len(msg), 10)

    def test_predict_without_model(self):
        """predict should return model_ready=False when no model is loaded."""
        from src.ai_detector import AIDetector
        import pathlib
        # Point to a non-existent model
        detector = AIDetector(model_path=pathlib.Path("models/nonexistent.pkl"))
        result = detector.predict("Some text to classify.")
        self.assertFalse(result["model_ready"])

    def test_predict_short_text_returns_result(self):
        """predict should return a dict even for very short text."""
        from src.ai_detector import AIDetector
        import pathlib
        detector = AIDetector(model_path=pathlib.Path("models/nonexistent.pkl"))
        result = detector.predict("Hi")
        self.assertIsInstance(result, dict)
        self.assertIn("prediction", result)


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(unittest.TestLoader().loadTestsFromModule(
        sys.modules[__name__]
    ))
    sys.exit(0 if result.wasSuccessful() else 1)

