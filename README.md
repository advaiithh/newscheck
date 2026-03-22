# Automated Fact-Checker for Vernacular News

## 1. Project Description
Misinformation in Indian social and vernacular news channels spreads faster than human fact-checkers can verify. This project builds a high-throughput fact-checking pipeline that strips noisy conversational text before retrieval and verification, which improves speed and reduces compute cost.

The system processes text posts (and optionally image posts through OCR), extracts factual claims, retrieves closest verified facts, and returns verdicts:
- True
- False
- Misleading

Core idea: optimize early, verify faster.

## 2. Problem Statement
Most AI fact-checking systems become expensive and slow when input text is noisy and long. Social media posts include emojis, clickbait, repeated words, and filler language that do not add factual meaning. This project solves that by reducing input size before claim extraction and retrieval.

## 3. Solution Overview
Pipeline stages:
1. Data Ingestion
2. Preprocessing Optimization
3. Claim Extraction
4. Fact Retrieval
5. Verification
6. Output with confidence

The optimization stage removes non-essential text so downstream components run faster with lower token usage.

## 4. Architecture Explanation
```
Input (Text/Image)
        |
        |--- if image ---> OCR (pytesseract, optional)
        v
Preprocessing Optimization
- Remove URLs, mentions, emojis
- Remove clickbait and filler words
- Normalize repeated words/characters
        v
Claim Extraction (lightweight heuristic)
        v
Fact Retrieval (tag-overlap retrieval, cache-enabled)
        v
Verification (verdict + confidence)
        v
Flag Misinformation Output
```

## 5. Technique Implementation: Pipeline Optimization
Implemented in preprocessing.py and integrated into main.py.

Optimization steps:
- URL and mention removal
- Emoji stripping
- Clickbait phrase removal (for example: "watch till end", "share this now")
- Repeated character normalization
- Filler-word filtering
- Repeated-word deduplication
- Per-post reduction metrics

Performance-focused techniques:
- Batch processing with ThreadPoolExecutor
- Retrieval caching via LRU cache
- Lightweight claim extraction (no heavy model dependency)
- Optional OCR path without changing the rest of the pipeline

## 6. Measurable Results (How to Report)
The benchmark reports:
- token_reduction_pct
- char_reduction_pct
- avg_latency_ms
- throughput_posts_per_min
- estimated_cost_before_usd
- estimated_cost_after_usd
- estimated_cost_savings_pct

Target outcomes for hackathon demo:
- 40-60% input reduction (tokens/chars)
- 2x faster processing vs non-optimized baseline
- under 200 ms average per-post pipeline latency (depends on machine)
- around 1000 posts/min in simulation with batching + threads
- around 80-85% demo accuracy on curated sample set

Note: accuracy depends on quality/size of verified fact database.

## 7. Real-World Feasibility
Why this can scale:
- Preprocessing sharply cuts unnecessary compute
- Stateless per-post processing supports horizontal scaling
- Caching handles repeated viral claims efficiently
- Can swap retrieval layer to FAISS or production vector DB without changing pipeline stages
- OCR path supports image-based misinformation (screenshots/posters)

### Critical Limitations (Be Honest in Viva)
- No real semantic understanding in this version:
  - No LLM reasoning
  - No semantic-embedding retrieval
- Cannot reliably handle:
  - Sarcasm
  - Indirect claims
  - Complex reasoning
- Performance depends heavily on:
  - Quality, coverage, and freshness of the fact database
- "Truth" is approximate:
  - Outputs are heuristic confidence-based verdicts, not authoritative truth
- Not equivalent to production-grade systems such as:
  - Google Fact Check
  - Advanced LLM reasoning systems

## 8. Demo Explanation (For Judges)
Suggested 2-3 minute demo flow:
1. Show noisy social post input (emoji-heavy + clickbait)
2. Show cleaned text and reduction percentage
3. Show extracted claim
4. Show retrieved verified fact and similarity score
5. Show final verdict + confidence
6. Run batch benchmark and display throughput/latency/cost savings
7. (Optional) Show image input through OCR and same pipeline output

One-line demo pitch:
"We reduce noisy input first, then verify faster and cheaper at scale while preserving factual context."

## 9. Judge-Facing Explanation
### Problem Understanding (10)
This system targets real misinformation volume in vernacular ecosystems where manual review cannot keep pace.

### Technique Implementation (25)
Pipeline optimization is the core technique: the preprocessing stage reduces non-factual noise before claim extraction and retrieval.

### Measurable Results (25)
The benchmark produces quantifiable latency, throughput, token reduction, and estimated cost savings.

### Real-World Feasibility (15)
Designed for modular deployment with scalable ingestion, cache-aware retrieval, and production-compatible retrieval backends.

### Demo & Reproducibility (15)
Single command demo with deterministic sample posts and benchmark output.

### Presentation & Clarity (10)
Simple architecture, focused modules, and clear result reporting.

## 10. Failure Learnings
- Large models increased latency and operational cost
- Noisy social text harmed retrieval quality
- Multilingual normalization is non-trivial and can introduce inconsistency
- Best results come from balancing fast rules with selective deeper verification

## 11. Future Improvements
- Better multilingual normalization and transliteration support
- Larger verified fact corpus with source trust scoring
- FAISS integration for larger-scale retrieval
- Temporal re-ranking to handle outdated facts
- More robust NLI-style verification for edge cases

## 12. Project Structure
- main.py: end-to-end pipeline, retrieval, verification, benchmark demo
- preprocessing.py: optimization stage and reduction metrics
- ml_classifier.py: train/evaluate TF-IDF + ML classifier and save model artifacts
- api.py: FastAPI endpoints for text/image/video prediction
- ui.py: Streamlit UI for text/image/video fact-check flow
- README.md: documentation and demo playbook

## 13. Setup and Run
### Prerequisites
- Python 3.9+

### Install
```bash
pip install -r requirements.txt
```

If OCR is not needed, you can skip these packages.

### Run
```bash
python main.py
```

This prints:
- per-post verdicts
- confidence and retrieval score
- benchmark metrics for speed, throughput, and cost savings

### Run API server (FastAPI)
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Open docs:
- http://127.0.0.1:8000/docs

Test predict endpoint:
```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "{\"text\":\"Breaking news: RBI is shutting all banks tomorrow\"}"
```

### Run UI (Streamlit)
```bash
streamlit run ui.py
```

UI features:
- Paste news text and check verdict
- Upload image and OCR + check verdict
- Upload video, sample frames with OCR, and check verdict

## 14. ML Fake-News Classification (Accuracy Track)
To demonstrate model-based fake-news accuracy, train a text classifier on your labeled dataset.

Expected CSV schema:
- text: news/post content
- label: target class (for example, Fake/Real)

Train command:
```bash
python ml_classifier.py --data data/fake_news.csv --text-col text --label-col label --model mlp
```

Supported model options:
- mlp
- linearsvc
- logreg
- randomforest

Model artifacts generated:
- artifacts/fake_news_model.joblib
- artifacts/fake_news_model_metrics.json

After training, run:
```bash
python main.py
```

The pipeline will automatically load the trained model and print ML predictions per post.

Accuracy note:
- Use stratified train/test split and avoid leakage to prevent unrealistically high scores.
- Very high accuracy (for example above 99%) can happen on easy/duplicate-heavy datasets; verify with clean splits.

## 15. Automatic Multi-Model Comparison (MLP vs SVC vs RF)
Use the comparison script to train on both dataset folders and produce a ranked accuracy table.

Data sources used automatically:
- News _dataset (Fake.csv, True.csv)
- FakeNewsData (class-wise text files like Fake/Satire folders)

Run:
```bash
python compare_models.py --csv-root "News _dataset" --txt-root "FakeNewsData"
```

Optional quick run (for faster debugging):
```bash
python compare_models.py --csv-root "News _dataset" --txt-root "FakeNewsData" --max-samples 3000
```

Output:
- Ranked table printed in terminal
- artifacts/model_comparison.csv
- artifacts/model_reports.joblib
- artifacts/best_fake_news_model.joblib

To use best model in pipeline, either:
- copy/rename artifacts/best_fake_news_model.joblib to artifacts/fake_news_model.joblib, or
- update model path in main.py if you prefer a different filename.

## 16. Docker
Build image:
```bash
docker build -t vernacular-fact-checker:latest .
```

Run container:
```bash
docker run --rm -p 8000:8000 vernacular-fact-checker:latest
```

Health check:
```bash
curl http://127.0.0.1:8000/health
```

## 17. Deploy to Railway (Exact Commands)
Install Railway CLI:
```bash
npm i -g @railway/cli
```

Login and initialize:
```bash
railway login
railway init
```

Deploy current project:
```bash
railway up
```

Open deployed app and verify:
```bash
railway open
railway domain
```

If needed, set environment variable explicitly:
```bash
railway variables set PORT=8000
```

## 18. Deploy to Render
This repo includes render.yaml for Blueprint deploy.

Push code:
```bash
git add .
git commit -m "Add FastAPI and Docker deploy"
git push
```

Deploy steps on Render:
1. Open Render dashboard.
2. New + -> Blueprint.
3. Select this repository.
4. Render reads render.yaml and deploys automatically.

Post-deploy checks:
- /health
- /docs

## 19. API Key Safety
Do not hardcode keys in source code or README.
Use environment variables for secrets.

Example:
```bash
setx FACTCHECK_API_KEY "your_new_key_here"
```

Then read in Python with:
```python
import os
api_key = os.getenv("FACTCHECK_API_KEY")
```

Rotate any key that was exposed publicly.
