import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from preprocessing import clean_text


@dataclass
class Record:
    text: str
    label: str
    source: str


def _normalize_label(raw_label: str) -> str:
    value = raw_label.lower()
    if "fake" in value or "satire" in value or "false" in value:
        return "Fake"
    if "true" in value or "real" in value or "genuine" in value:
        return "True"
    return "Unknown"


def _load_csv_source(root: Path) -> List[Record]:
    records: List[Record] = []
    if not root.exists():
        return records

    for csv_path in root.glob("*.csv"):
        file_label = _normalize_label(csv_path.stem)
        if file_label == "Unknown":
            continue

        df = pd.read_csv(csv_path)
        text_col = "text" if "text" in df.columns else None
        if text_col is None and "title" in df.columns:
            text_col = "title"
        if text_col is None:
            continue

        for text in df[text_col].dropna().astype(str):
            cleaned = clean_text(text)
            if len(cleaned.split()) < 4:
                continue
            records.append(Record(text=cleaned, label=file_label, source=str(csv_path.name)))

    return records


def _load_txt_source(root: Path) -> List[Record]:
    records: List[Record] = []
    if not root.exists():
        return records

    txt_files = list(root.rglob("*.txt"))
    for path in txt_files:
        path_label = _normalize_label(str(path.parent))
        if path_label == "Unknown":
            continue

        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        cleaned = clean_text(content)
        if len(cleaned.split()) < 4:
            continue

        source = str(path.relative_to(root))
        records.append(Record(text=cleaned, label=path_label, source=source))

    return records


def load_combined_dataset(csv_root: str, txt_root: str, max_samples: int = 0) -> pd.DataFrame:
    all_records = _load_csv_source(Path(csv_root)) + _load_txt_source(Path(txt_root))
    df = pd.DataFrame([r.__dict__ for r in all_records])

    if df.empty:
        raise ValueError("No training records found in provided dataset folders.")

    df = df[df["label"].isin(["Fake", "True"])].drop_duplicates(subset=["text"]).reset_index(drop=True)
    if max_samples > 0 and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
    return df


def build_models() -> Dict[str, object]:
    return {
        "MLPClassifier": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=220, random_state=42),
        "LinearSVC": LinearSVC(),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),
    }


def evaluate_models(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, Dict[str, Dict[str, str]], Dict[str, object]]:
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=test_size,
        random_state=42,
        stratify=df["label"],
    )

    rows = []
    reports: Dict[str, Dict[str, str]] = {}
    trained_pipelines: Dict[str, object] = {}

    for model_name, model in build_models().items():
        pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        min_df=2,
                        max_df=0.95,
                        sublinear_tf=True,
                        max_features=40000,
                        dtype=np.float32,
                    ),
                ),
                ("clf", model),
            ]
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        acc = accuracy_score(y_test, predictions)

        rows.append(
            {
                "Model": model_name,
                "Accuracy": round(float(acc) * 100.0, 4),
            }
        )
        reports[model_name] = classification_report(y_test, predictions, output_dict=True)
        trained_pipelines[model_name] = pipeline

    ranked = pd.DataFrame(rows).sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
    ranked.index = ranked.index + 1
    ranked.index.name = "Rank"
    return ranked, reports, trained_pipelines


def print_ranked_table(ranked_df: pd.DataFrame) -> None:
    print("\n=== Model Comparison (Ranked by Accuracy) ===")
    header = f"{'Rank':<6}{'Model':<28}{'Accuracy':>12}"
    print(header)
    print("-" * len(header))
    for rank, row in ranked_df.iterrows():
        print(f"{rank:<6}{row['Model']:<28}{row['Accuracy']:>11.4f}%")


def run() -> None:
    parser = argparse.ArgumentParser(description="Compare MLP vs SVC vs RF for fake-news classification")
    parser.add_argument("--csv-root", default="News _dataset", help="Folder containing Fake.csv/True.csv")
    parser.add_argument("--txt-root", default="FakeNewsData", help="Folder containing class-wise text files")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=20000,
        help="Optional cap on total samples (use 0 to disable cap)",
    )
    parser.add_argument(
        "--save-dir",
        default="artifacts",
        help="Directory to save ranked table and best model",
    )
    args = parser.parse_args()

    df = load_combined_dataset(args.csv_root, args.txt_root, args.max_samples)

    print("=== Dataset Summary ===")
    print(f"Total samples: {len(df)}")
    print("Class distribution:")
    print(df["label"].value_counts())

    ranked, reports, pipelines = evaluate_models(df, test_size=args.test_size)
    print_ranked_table(ranked)

    os.makedirs(args.save_dir, exist_ok=True)
    ranked_path = Path(args.save_dir) / "model_comparison.csv"
    report_path = Path(args.save_dir) / "model_reports.joblib"
    best_name = ranked.iloc[0]["Model"]
    best_model_path = Path(args.save_dir) / "best_fake_news_model.joblib"

    ranked.to_csv(ranked_path)
    joblib.dump(reports, report_path)
    joblib.dump(pipelines[best_name], best_model_path)

    print("\n=== Saved Artifacts ===")
    print(f"Ranked table: {ranked_path}")
    print(f"All reports: {report_path}")
    print(f"Best model ({best_name}): {best_model_path}")


if __name__ == "__main__":
    run()
