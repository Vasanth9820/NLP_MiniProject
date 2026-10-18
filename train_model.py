"""
train_model.py
--------------
Training script for the AI-content classifier.

Usage:
    python train_model.py

The script:
  1. Loads data/ai_text_dataset.csv  (columns: text, label)
  2. Validates columns
  3. Preprocesses text
  4. Splits into train/test sets
  5. Trains a TF-IDF + Logistic Regression pipeline
  6. Evaluates with accuracy, precision, recall, F1
  7. Saves the model to models/ai_detector.pkl

Dataset format
--------------
CSV with two columns:
    text   — the document/passage text
    label  — 'ai' or 'human'

The minimum recommended dataset size is ~200 examples per class.
The demo dataset included is intentionally small and is for testing only.
"""

from __future__ import annotations

import pathlib
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure src/ is importable
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.preprocessing import preprocess_text  # noqa: E402

# Paths
DATA_PATH = ROOT / "data" / "ai_text_dataset.csv"
MODEL_PATH = ROOT / "models" / "ai_detector.pkl"
MODEL_PATH.parent.mkdir(exist_ok=True)

MIN_SAMPLES_PER_CLASS = 10  # Warn below this count


def load_and_validate(path: pathlib.Path) -> pd.DataFrame:
    """Load CSV dataset and validate required columns."""
    print(f"[*] Loading dataset from: {path}")
    if not path.exists():
        print(f"[ERROR] Dataset not found at {path}")
        print(
            "   Create a CSV file with columns 'text' and 'label' (values: 'ai', 'human')."
        )
        sys.exit(1)

    df = pd.read_csv(path)

    required = {"text", "label"}
    missing = required - set(df.columns.str.lower())
    if missing:
        print(f"[ERROR] Missing columns: {missing}. Dataset must have 'text' and 'label'.")
        sys.exit(1)

    # Normalise column names
    df.columns = [c.lower() for c in df.columns]
    df = df[["text", "label"]].dropna()
    df["label"] = df["label"].str.lower().str.strip()

    valid_labels = {"ai", "human"}
    invalid = set(df["label"].unique()) - valid_labels
    if invalid:
        print(f"[ERROR] Invalid labels found: {invalid}. Labels must be 'ai' or 'human'.")
        sys.exit(1)

    counts = df["label"].value_counts()
    print("\n[INFO] Label distribution:")
    print(counts.to_string())

    for label in valid_labels:
        n = counts.get(label, 0)
        if n < MIN_SAMPLES_PER_CLASS:
            print(
                f"\n[WARN]  Only {n} '{label}' sample(s) found. "
                f"Minimum {MIN_SAMPLES_PER_CLASS} per class recommended. "
                f"Results may be unreliable."
            )

    return df


def preprocess_column(texts: pd.Series) -> list[str]:
    """Apply the preprocessing pipeline to each text entry."""
    print("\n[*] Preprocessing text...")
    processed = []
    for i, t in enumerate(texts):
        processed.append(preprocess_text(str(t)))
        if (i + 1) % 100 == 0:
            print(f"   Processed {i + 1}/{len(texts)} samples...")
    return processed


def build_pipeline() -> Pipeline:
    """Create the TF-IDF + Logistic Regression pipeline."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 3),
                    max_features=20_000,
                    sublinear_tf=True,
                    min_df=1,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    C=1.0,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )


def train_and_evaluate(df: pd.DataFrame) -> Pipeline:
    """Train the pipeline and print evaluation metrics."""
    X = preprocess_column(df["text"])
    y = df["label"].tolist()

    test_size = 0.2 if len(df) >= 10 else 0.0

    if test_size > 0:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        print(f"\n[*] Train/Test split: {len(X_train)} train / {len(X_test)} test")
    else:
        X_train, y_train = X, y
        X_test, y_test = [], []
        print("\n[WARN] Dataset too small for train/test split. Training on all data.")

    print("\n[*] Training classifier...")
    t0 = time.time()
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"   Training complete in {elapsed:.2f}s")

    if X_test:
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"\n[OK] Test Accuracy: {acc * 100:.2f}%")
        print("\n[INFO] Classification Report:")
        print(classification_report(y_test, y_pred, target_names=["ai", "human"]))

    return pipeline


def save_model(pipeline: Pipeline, path: pathlib.Path) -> None:
    """Save the trained pipeline to disk with joblib."""
    joblib.dump(pipeline, path)
    print(f"\n[OK] Model saved to: {path}")


def main() -> None:
    print("=" * 60)
    print("  AI Content Classifier -- Training Script")
    print("=" * 60)

    df = load_and_validate(DATA_PATH)
    pipeline = train_and_evaluate(df)
    save_model(pipeline, MODEL_PATH)

    print("\n[DONE] Training complete! Start the application with:")
    print("   streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
