import os
import time
import importlib.util
import importlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Sequence

from preprocessing import clean_text, reduction_stats

OCR_AVAILABLE = bool(importlib.util.find_spec("pytesseract")) and bool(
    importlib.util.find_spec("PIL")
)


@dataclass
class VerifiedFact:
    fact_id: str
    text: str
    verdict: str
    tags: Sequence[str]


VERIFIED_FACTS = [
    VerifiedFact(
        fact_id="F001",
        text="No nationwide bank closure was announced by RBI in 2026.",
        verdict="False",
        tags=["rbi", "bank", "closure", "nationwide", "2026"],
    ),
    VerifiedFact(
        fact_id="F002",
        text="India has no policy that gives every citizen 5000 rupees per day.",
        verdict="False",
        tags=["india", "policy", "citizen", "5000", "per", "day"],
    ),
    VerifiedFact(
        fact_id="F003",
        text="Heatwaves can happen in March in multiple Indian states.",
        verdict="True",
        tags=["heatwave", "march", "indian", "states", "weather"],
    ),
    VerifiedFact(
        fact_id="F004",
        text="The Election Commission publishes official polling schedules on its portal.",
        verdict="True",
        tags=["election", "commission", "official", "schedule", "portal"],
    ),
    VerifiedFact(
        fact_id="F005",
        text="Government schemes are announced through official notifications, not random forwards.",
        verdict="Misleading",
        tags=["government", "scheme", "official", "notification", "forward"],
    ),
]


@lru_cache(maxsize=1)
def _load_ml_model(model_path: str):
    candidate_paths = [
        model_path,
        "artifacts/best_fake_news_model.joblib",
        "artifacts/fake_news_model.joblib",
    ]
    resolved_path = next((path for path in candidate_paths if os.path.exists(path)), None)
    if resolved_path is None:
        return None
    joblib = importlib.import_module("joblib")
    return joblib.load(resolved_path)


def classify_fake_news_ml(text: str, model_path: str = "artifacts/fake_news_model.joblib") -> Dict[str, object]:
    model = _load_ml_model(model_path)
    if model is None:
        return {
            "available": False,
            "prediction": "N/A",
            "model_path": model_path,
        }

    prediction = model.predict([clean_text(text)])[0]
    return {
        "available": True,
        "prediction": str(prediction),
        "model_path": model_path,
    }


def extract_text_from_image(image_path: str) -> str:
    if not OCR_AVAILABLE:
        raise RuntimeError("OCR dependencies not installed. Install pytesseract and pillow.")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    pytesseract = importlib.import_module("pytesseract")
    pil_image = importlib.import_module("PIL.Image")
    return pytesseract.image_to_string(pil_image.open(image_path))


def claim_extraction(cleaned_text: str) -> str:
    # Keep claim extraction lightweight for throughput.
    sentences = [segment.strip() for segment in cleaned_text.split(".") if segment.strip()]
    if sentences:
        return max(sentences, key=lambda s: len(s.split()))
    return cleaned_text


def _overlap_score(claim_tokens: Sequence[str], fact_tags: Sequence[str]) -> float:
    claim_set = set(claim_tokens)
    fact_set = set(fact_tags)
    if not claim_set or not fact_set:
        return 0.0
    overlap = len(claim_set.intersection(fact_set))
    return overlap / len(fact_set)


@lru_cache(maxsize=2048)
def retrieve_fact_for_claim(claim: str) -> Dict[str, object]:
    claim_tokens = claim.split()
    scored = []
    for fact in VERIFIED_FACTS:
        score = _overlap_score(claim_tokens, fact.tags)
        scored.append((score, fact))

    best_score, best_fact = max(scored, key=lambda x: x[0])
    return {
        "fact_id": best_fact.fact_id,
        "fact_text": best_fact.text,
        "base_verdict": best_fact.verdict,
        "retrieval_score": round(best_score, 3),
    }


def verify_claim_against_fact(claim: str, retrieved: Dict[str, object]) -> Dict[str, object]:
    score = float(retrieved["retrieval_score"])
    verdict = str(retrieved["base_verdict"])

    if score < 0.2:
        verdict = "Misleading"
    confidence = min(0.99, max(0.35, 0.45 + score))

    return {
        "verdict": verdict,
        "confidence": round(confidence, 3),
        "matched_fact_id": retrieved["fact_id"],
        "matched_fact": retrieved["fact_text"],
        "retrieval_score": score,
    }


def process_post(input_data: str, is_image: bool = False) -> Dict[str, object]:
    raw_text = extract_text_from_image(input_data) if is_image else input_data
    cleaned = clean_text(raw_text)
    claim = claim_extraction(cleaned)
    retrieved = retrieve_fact_for_claim(claim)
    result = verify_claim_against_fact(claim, retrieved)
    ml_result = classify_fake_news_ml(raw_text)

    return {
        "input": input_data,
        "raw_text": raw_text,
        "cleaned_text": cleaned,
        "claim": claim,
        "preprocessing_metrics": reduction_stats(raw_text, cleaned),
        "verification": result,
        "ml_classification": ml_result,
    }


def process_batch(posts: Sequence[str], workers: int = 8) -> List[Dict[str, object]]:
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(process_post, posts))
    elapsed = time.perf_counter() - start

    for item in results:
        item["pipeline_latency_ms"] = round((elapsed / max(1, len(posts))) * 1000.0, 2)
    return results


def benchmark_pipeline(sample_posts: Sequence[str], runs: int = 3) -> Dict[str, float]:
    total_chars_before = 0
    total_chars_after = 0
    total_tokens_before = 0
    total_tokens_after = 0
    total_elapsed = 0.0

    for _ in range(runs):
        start = time.perf_counter()
        results = process_batch(sample_posts)
        total_elapsed += time.perf_counter() - start

        for result in results:
            metrics = result["preprocessing_metrics"]
            total_chars_before += int(metrics["original_chars"])
            total_chars_after += int(metrics["cleaned_chars"])
            total_tokens_before += int(metrics["original_tokens"])
            total_tokens_after += int(metrics["cleaned_tokens"])

    processed_items = len(sample_posts) * runs
    avg_latency_ms = (total_elapsed / max(1, processed_items)) * 1000.0
    throughput_per_min = (processed_items / max(1e-9, total_elapsed)) * 60.0
    token_reduction_pct = (
        ((total_tokens_before - total_tokens_after) / max(1, total_tokens_before)) * 100.0
    )
    char_reduction_pct = (
        ((total_chars_before - total_chars_after) / max(1, total_chars_before)) * 100.0
    )

    # Cost simulation: token-reduction directly lowers LLM/API spend.
    base_cost = total_tokens_before * 0.000002
    optimized_cost = total_tokens_after * 0.000002

    return {
        "items_processed": float(processed_items),
        "avg_latency_ms": round(avg_latency_ms, 2),
        "throughput_posts_per_min": round(throughput_per_min, 2),
        "token_reduction_pct": round(token_reduction_pct, 2),
        "char_reduction_pct": round(char_reduction_pct, 2),
        "estimated_cost_before_usd": round(base_cost, 4),
        "estimated_cost_after_usd": round(optimized_cost, 4),
        "estimated_cost_savings_pct": round(
            ((base_cost - optimized_cost) / max(1e-9, base_cost)) * 100.0,
            2,
        ),
    }


def demo() -> None:
    sample_posts = [
        "BREAKING!!! Share this now!!! Every citizen gets 5000 rupees daily from tonight 😱😱",
        "Election dates out now, check official Election Commission website for final schedule.",
        "Forward this viral update: RBI shutting all banks nationwide tomorrow!!!",
        "Heatwave alert in north India this week. Follow IMD advisories.",
        "OMG watch till end this shocking policy update wow wow wow",
    ]

    results = process_batch(sample_posts)
    summary = benchmark_pipeline(sample_posts, runs=5)

    print("=== Single Batch Results ===")
    for i, item in enumerate(results, start=1):
        verification = item["verification"]
        ml_prediction = item["ml_classification"]
        print(f"{i}. Claim: {item['claim']}")
        print(
            f"   Verdict: {verification['verdict']} | Confidence: {verification['confidence']} | "
            f"Retrieval Score: {verification['retrieval_score']}"
        )
        if ml_prediction["available"]:
            print(f"   ML Prediction: {ml_prediction['prediction']}")
        else:
            print("   ML Prediction: Model not trained yet")

    print("\n=== Benchmark Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    demo()
