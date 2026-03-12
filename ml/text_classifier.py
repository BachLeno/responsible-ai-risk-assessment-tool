from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Resolve project root robustly
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_PATH = PROJECT_ROOT / "ml" / "model.joblib"
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "training_data.csv"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "ml" / "metrics.json"


@dataclass
class MLResult:
    predicted_label: str
    confidence: float
    class_probabilities: List[Tuple[str, float]]


def _build_pipeline() -> Pipeline:
    """
    Create the text classification pipeline (TF-IDF + Logistic Regression).
    """
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    max_features=5000,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def _load_training_data(train_csv_path: Path) -> tuple[list[str], list[str]]:
    """
    Load training data from CSV (expects columns: text, label).
    """
    if not train_csv_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_csv_path}")

    df = pd.read_csv(train_csv_path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("training_data.csv must contain columns: text,label")

    X = df["text"].astype(str).tolist()
    y = df["label"].astype(str).tolist()

    if len(X) < 10:
        raise ValueError(
            f"Training data too small ({len(X)} rows). "
            "Expand dataset for proper evaluation."
        )

    return X, y


def train_and_save_model(
    train_csv_path: Path = DEFAULT_TRAIN_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> None:
    """
    Train the classifier on the full dataset and save the model.
    """
    X, y = _load_training_data(train_csv_path)

    pipeline = _build_pipeline()
    pipeline.fit(X, y)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)


def evaluate_model(
    train_csv_path: Path = DEFAULT_TRAIN_PATH,
    test_size: float = 0.25,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Evaluate the model using a stratified train/test split.

    Returns:
        - Accuracy
        - Macro F1 score
        - Confusion matrix
        - Classification report
    """
    X, y = _load_training_data(train_csv_path)

    # Keep label distribution similar in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    pipeline = _build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    labels_sorted = sorted(list(set(y)))
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)

    metrics: Dict[str, Any] = {
        "data": {
            "train_csv_path": "data/training_data.csv",
            "rows_total": len(X),
            "test_size": test_size,
            "random_state": random_state,
            "labels": labels_sorted,
        },
        "metrics": {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        },
        "confusion_matrix": {
            "labels": labels_sorted,
            "matrix": cm.tolist(),
        },
        "classification_report": classification_report(
            y_test,
            y_pred,
            labels=labels_sorted,
            output_dict=True,
            zero_division=0,
        ),
    }

    return metrics


def train_evaluate_and_save(
    train_csv_path: Path = DEFAULT_TRAIN_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    metrics_path: Path = DEFAULT_METRICS_PATH,
    test_size: float = 0.25,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Train the model, evaluate it, and save both model and metrics.
    """
    train_and_save_model(train_csv_path=train_csv_path, model_path=model_path)

    metrics = evaluate_model(
        train_csv_path=train_csv_path,
        test_size=test_size,
        random_state=random_state,
    )

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics


def _load_model(model_path: Path = DEFAULT_MODEL_PATH) -> Pipeline:
    """
    Load a saved model from disk.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Train it first with: python -m ml.text_classifier train"
        )
    return joblib.load(model_path)


def predict_risk_from_text(text: str, model_path: Path = DEFAULT_MODEL_PATH) -> MLResult:
    """
    Predict a risk label from free-text input.
    """
    model = _load_model(model_path)

    probs = model.predict_proba([text])[0]
    classes = list(model.classes_)

    paired = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
    predicted_label = str(paired[0][0])
    confidence = float(paired[0][1])

    return MLResult(
        predicted_label=predicted_label,
        confidence=confidence,
        class_probabilities=[(str(c), float(p)) for c, p in paired],
    )


# CLI usage
if __name__ == "__main__":
    import sys

    cmd = sys.argv[1].lower().strip() if len(sys.argv) >= 2 else ""

    if cmd == "train":
        train_and_save_model()
        print(f"Model trained and saved to: {DEFAULT_MODEL_PATH}")

    elif cmd == "eval":
        metrics = evaluate_model()
        print(json.dumps(metrics, indent=2))

    elif cmd in {"train-eval", "traineval", "train_eval"}:
        metrics = train_evaluate_and_save()
        print(f"Model trained and saved to: {DEFAULT_MODEL_PATH}")
        print(f"Metrics saved to: {DEFAULT_METRICS_PATH}")
        print(json.dumps(metrics, indent=2))

    else:
        print("Usage:")
        print("  python -m ml.text_classifier train")
        print("  python -m ml.text_classifier eval")
        print("  python -m ml.text_classifier train-eval")