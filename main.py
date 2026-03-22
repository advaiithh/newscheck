import os
import time
import importlib.util
import importlib
import json
import re
import urllib.parse
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Sequence

from preprocessing import clean_text, reduction_stats

OCR_AVAILABLE = bool(importlib.util.find_spec("pytesseract")) and bool(
    importlib.util.find_spec("PIL")
)
VIDEO_OCR_AVAILABLE = OCR_AVAILABLE and bool(importlib.util.find_spec("cv2"))

SYSTEM_LIMITATIONS = {
    "no_real_understanding": [
        "No LLM reasoning in the verification pipeline.",
        "No semantic-embedding retrieval is used.",
    ],
    "cannot_handle_reliably": [
        "sarcasm",
        "indirect claims",
        "complex reasoning",
    ],
    "dependency_risks": [
        "Output quality depends heavily on verified fact-database quality, coverage, and freshness.",
    ],
    "truth_scope": [
        "Verdicts are approximate heuristic outputs, not authoritative truth guarantees.",
        "Not equivalent to production-grade systems such as Google Fact Check or advanced LLM reasoning systems.",
    ],
}


def get_system_limitations() -> Dict[str, List[str]]:
    return SYSTEM_LIMITATIONS


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
    try:
        pytesseract.get_tesseract_version()
    except Exception as exc:
        raise RuntimeError("Tesseract OCR engine is not installed or not available in PATH.") from exc
    return pytesseract.image_to_string(pil_image.open(image_path))


def extract_text_from_video(video_path: str, sample_every_n_frames: int = 30, max_frames: int = 20) -> str:
    if not VIDEO_OCR_AVAILABLE:
        raise RuntimeError("Video OCR dependencies not installed. Install opencv-python-headless, pytesseract and pillow.")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cv2 = importlib.import_module("cv2")
    pytesseract = importlib.import_module("pytesseract")
    pil_image = importlib.import_module("PIL.Image")
    try:
        pytesseract.get_tesseract_version()
    except Exception as exc:
        raise RuntimeError("Tesseract OCR engine is not installed or not available in PATH.") from exc

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video file: {video_path}")

    frame_texts: List[str] = []
    frame_index = 0
    sampled = 0
    try:
        while sampled < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % max(1, sample_every_n_frames) == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                text = pytesseract.image_to_string(pil_image.fromarray(rgb)).strip()
                if text:
                    frame_texts.append(text)
                sampled += 1
            frame_index += 1
    finally:
        cap.release()

    merged = " ".join(frame_texts).strip()
    if not merged:
        raise RuntimeError("No readable text detected in video frames.")
    return merged


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
    overlap_tokens = len(set(claim_tokens).intersection(set(best_fact.tags)))
    return {
        "fact_id": best_fact.fact_id,
        "fact_text": best_fact.text,
        "base_verdict": best_fact.verdict,
        "retrieval_score": round(best_score, 3),
        "overlap_tokens": overlap_tokens,
    }


def verify_claim_against_fact(claim: str, retrieved: Dict[str, object]) -> Dict[str, object]:
    score = float(retrieved["retrieval_score"])
    overlap_tokens = int(retrieved.get("overlap_tokens", 0))
    verdict = str(retrieved["base_verdict"])

    # Keep local fact-store verdict conservative to reduce false positives.
    if score < 0.6 or overlap_tokens < 3:
        verdict = "Unverified"
    confidence = 0.4 if verdict == "Unverified" else min(0.99, max(0.6, 0.55 + score * 0.4))

    return {
        "verdict": verdict,
        "confidence": round(confidence, 3),
        "matched_fact_id": retrieved["fact_id"],
        "matched_fact": retrieved["fact_text"],
        "retrieval_score": score,
        "overlap_tokens": overlap_tokens,
    }


def _map_textual_rating_to_verdict(textual_rating: str) -> str:
    normalized = textual_rating.strip().lower()
    if any(token in normalized for token in ("false", "fake", "hoax", "pants on fire", "scam")):
        return "False"
    if any(token in normalized for token in ("mostly false", "partly false", "half true", "mixed", "misleading")):
        return "Misleading"
    if any(token in normalized for token in ("true", "correct", "accurate", "mostly true")):
        return "True"
    return "Unverified"


def _tokenize_for_match(text: str) -> List[str]:
    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 1]


def _claim_query_variants(claim: str) -> List[str]:
    tokens = _tokenize_for_match(claim)
    variants: List[str] = []
    if claim.strip():
        variants.append(claim.strip())

    if len(tokens) > 12:
        variants.append(" ".join(tokens[:12]))
    if len(tokens) > 6:
        variants.append(" ".join(tokens[-10:]))

    stopwords = {
        "the", "is", "are", "was", "were", "will", "would", "can", "could", "a", "an", "and",
        "or", "to", "for", "of", "in", "on", "at", "from", "with", "by", "this", "that",
    }
    keyword_tokens = [token for token in tokens if token not in stopwords]
    if keyword_tokens:
        variants.append(" ".join(keyword_tokens[:12]))

    deduped: List[str] = []
    seen = set()
    for variant in variants:
        normalized = variant.strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(variant.strip())
    return deduped[:3]


def _match_overlap_score(claim_text: str, candidate_text: str, review_title: str) -> float:
    claim_tokens = set(_tokenize_for_match(claim_text))
    candidate_tokens = set(_tokenize_for_match(f"{candidate_text} {review_title}"))
    if not claim_tokens or not candidate_tokens:
        return 0.0
    return len(claim_tokens.intersection(candidate_tokens)) / len(claim_tokens)


@lru_cache(maxsize=1024)
def verify_claim_with_real_world_source(claim: str) -> Dict[str, object]:
    api_key = os.getenv("FACTCHECK_API_KEY", "").strip()
    if not api_key:
        return {"available": False, "reason": "missing_api_key"}

    query_variants = _claim_query_variants(claim)
    all_claims: List[Dict[str, object]] = []
    for variant in query_variants:
        query = urllib.parse.quote(variant)
        url = (
            "https://factchecktools.googleapis.com/v1alpha1/claims:search"
            f"?query={query}&languageCode=en&key={api_key}"
        )
        try:
            with urllib.request.urlopen(url, timeout=8) as response:
                payload = json.loads(response.read().decode("utf-8"))
            all_claims.extend(payload.get("claims", []))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            return {"available": False, "reason": f"request_failed: {exc}"}

    if not all_claims:
        return {"available": True, "matched": False}

    best_match = None
    best_score = 0.0
    for candidate in all_claims:
        candidate_text = str(candidate.get("text", "")).strip()
        reviews = candidate.get("claimReview", [])
        if not reviews:
            score = _match_overlap_score(claim, candidate_text, "")
            if score > best_score:
                best_score = score
                best_match = (candidate, {})
            continue
        for review in reviews:
            review_title = str(review.get("title", "")).strip()
            score = _match_overlap_score(claim, candidate_text, review_title)
            if score > best_score:
                best_score = score
                best_match = (candidate, review)

    if best_match is None or best_score < 0.2:
        return {
            "available": True,
            "matched": False,
            "reason": "no_high_similarity_match",
            "best_match_score": round(best_score, 3),
        }

    top_claim, top_review = best_match
    textual_rating = str(top_review.get("textualRating", "")).strip()
    mapped_verdict = _map_textual_rating_to_verdict(textual_rating)

    return {
        "available": True,
        "matched": True,
        "source": "Google Fact Check Tools API",
        "best_match_score": round(best_score, 3),
        "claim_text": top_claim.get("text", ""),
        "claimant": top_claim.get("claimant", ""),
        "review_title": top_review.get("title", ""),
        "publisher": (top_review.get("publisher") or {}).get("name", ""),
        "review_url": top_review.get("url", ""),
        "review_date": top_review.get("reviewDate", ""),
        "textual_rating": textual_rating or "N/A",
        "mapped_verdict": mapped_verdict,
    }


def process_post(input_data: str, is_image: bool = False) -> Dict[str, object]:
    raw_text = extract_text_from_image(input_data) if is_image else input_data
    cleaned = clean_text(raw_text)
    claim = claim_extraction(cleaned)
    retrieved = retrieve_fact_for_claim(claim)
    local_result = verify_claim_against_fact(claim, retrieved)
    real_world_result = verify_claim_with_real_world_source(claim)
    result = dict(local_result)
    if real_world_result.get("available") and real_world_result.get("matched"):
        result["verdict"] = real_world_result.get("mapped_verdict", "Unverified")
        result["confidence"] = 0.9 if result["verdict"] != "Unverified" else 0.5
        result["evidence_source"] = real_world_result.get("source", "external")
    else:
        result["evidence_source"] = "local_fact_store"
    ml_result = classify_fake_news_ml(raw_text)

    return {
        "input": input_data,
        "raw_text": raw_text,
        "cleaned_text": cleaned,
        "claim": claim,
        "preprocessing_metrics": reduction_stats(raw_text, cleaned),
        "verification": result,
        "real_world_verification": real_world_result,
        "ml_classification": ml_result,
        "system_limitations": get_system_limitations(),
    }


def process_video_post(video_path: str) -> Dict[str, object]:
    extracted_text = extract_text_from_video(video_path)
    result = process_post(extracted_text, is_image=False)
    result["input_video"] = video_path
    return result


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
