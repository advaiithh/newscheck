import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from preprocessing import clean_text


@dataclass
class TrainingConfig:
    data_path: str
    text_col: str = "text"
    label_col: str = "label"
    model_name: str = "mlp"
    test_size: float = 0.2
    random_state: int = 42
    save_path: str = "artifacts/fake_news_model.joblib"


def _get_model(model_name: str):
    name = model_name.lower()
    if name == "mlp":
        return MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=350, random_state=42)
    if name == "linearsvc":
        return LinearSVC()
    if name == "logreg":
        return LogisticRegression(max_iter=1200)
    if name == "randomforest":
        return RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    raise ValueError(
        "Unsupported model_name. Use one of: mlp, linearsvc, logreg, randomforest"
    )


def _load_dataset(path: str, text_col: str, label_col: str) -> Tuple[pd.Series, pd.Series]:
    df = pd.read_csv(path)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(
            f"CSV must contain columns '{text_col}' and '{label_col}'. Found: {list(df.columns)}"
        )

    df = df[[text_col, label_col]].dropna()
    df[text_col] = df[text_col].astype(str).map(clean_text)
    return df[text_col], df[label_col].astype(str)


def train_and_evaluate(config: TrainingConfig) -> Dict[str, object]:
    X, y = _load_dataset(config.data_path, config.text_col, config.label_col)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )

    model = _get_model(config.model_name)
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            ("clf", model),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    os.makedirs(os.path.dirname(config.save_path), exist_ok=True)
    joblib.dump(pipeline, config.save_path)

    metrics = {
        "model_name": config.model_name,
        "accuracy": round(float(accuracy), 6),
        "test_size": config.test_size,
        "samples_total": int(len(X)),
        "samples_train": int(len(X_train)),
        "samples_test": int(len(X_test)),
        "save_path": config.save_path,
        "classification_report": report,
    }

    metrics_path = config.save_path.replace(".joblib", "_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=True, indent=2)

    return metrics


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="Train fake-news text classifier")
    parser.add_argument("--data", required=True, help="Path to CSV dataset")
    parser.add_argument("--text-col", default="text", help="Text column name")
    parser.add_argument("--label-col", default="label", help="Label column name")
    parser.add_argument(
        "--model",
        default="mlp",
        choices=["mlp", "linearsvc", "logreg", "randomforest"],
        help="Model choice",
    )
    parser.add_argument("--save", default="artifacts/fake_news_model.joblib", help="Model output path")
    args = parser.parse_args()

    config = TrainingConfig(
        data_path=args.data,
        text_col=args.text_col,
        label_col=args.label_col,
        model_name=args.model,
        save_path=args.save,
    )
    metrics = train_and_evaluate(config)

    print("=== Training Complete ===")
    print(f"Model: {metrics['model_name']}")
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"Saved model: {metrics['save_path']}")


if __name__ == "__main__":
    run_cli()
