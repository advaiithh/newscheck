# 24-Hour Execution Guide (Internship-Ready)

## Goal by Tomorrow
Ship a clean, demo-ready fake-news fact-checking project with:
- working API
- reproducible training + inference
- measurable metrics
- clear documentation
- deployment proof (or deployment-ready package)

This plan is optimized for a 3-member team:
- hamsi
- advaith
- you

## What "Internship-Ready" Means for This Project
By submission time, you should show:
1. Clear architecture and modular code
2. Reproducible setup commands
3. Measurable ML and pipeline metrics
4. API endpoints that work reliably
5. Clean README and project guide
6. Demo script that runs without surprises

## Mandatory Deliverables
1. Codebase with no obvious runtime errors
2. Trained model artifact in artifacts/
3. Model comparison table (MLP vs LinearSVC vs RF)
4. API working locally (/health, /predict, /predict-batch)
5. Updated README with run + deploy steps
6. 2-3 minute demo flow with sample inputs and outputs

## Team Split (Final)

### Member 1: hamsi (ML + Evaluation Owner)
Primary responsibility: model quality and measurable results.

Tasks:
1. Run model comparison using both datasets:
   - .\\.venv\\Scripts\\python.exe compare_models.py --csv-root "News _dataset" --txt-root "FakeNewsData"
2. Save and verify artifacts:
   - artifacts/model_comparison.csv
   - artifacts/model_reports.joblib
   - artifacts/best_fake_news_model.joblib
3. Record final metrics in README:
   - ranked accuracy table
   - class distribution
   - sample size used
4. Validate inference path uses best model by running:
   - .\\.venv\\Scripts\\python.exe main.py
5. Add one short "limitations" note with leakage caution and class-balance note.

Definition of done:
- Ranked table present
- Metrics copied into README
- Inference runs with trained model

### Member 2: advaith (API + Deployment Owner)
Primary responsibility: backend reliability and deployment readiness.

Tasks:
1. Run API locally:
   - .\\.venv\\Scripts\\python.exe -m uvicorn api:app --host 0.0.0.0 --port 8000
2. Test endpoints manually from docs and curl:
   - GET /health
   - POST /predict
   - POST /predict-batch
   - POST /predict-image (if OCR setup available)
3. Deployment track A (no Docker, faster):
   - Deploy on Railway with Python runtime (if Docker unavailable)
4. Deployment track B (Docker optional if available):
   - Build and run image locally
5. Add final deployed URL (or clear blocker + workaround) in README.

Definition of done:
- API endpoints verified
- At least one deployment path ready
- URL/shared proof ready for demo

### Member 3: you (Product + Documentation + Demo Owner)
Primary responsibility: presentation quality and final integration.

Tasks:
1. Finalize README storytelling:
   - problem statement
   - architecture
   - optimization strategy
   - measurable results
   - real-world feasibility
2. Prepare final demo script (2-3 minutes):
   - noisy input -> cleaned text -> claim -> retrieval -> verdict -> ML prediction -> metrics
3. Create 5 curated demo inputs:
   - 2 obvious fake
   - 2 likely true
   - 1 ambiguous/misleading
4. Run full dry-run:
   - train/compare once
   - run API
   - test predict request live
5. Final quality gate:
   - no secrets in code
   - commands in README work
   - all important outputs captured as screenshots

Definition of done:
- README polished
- Demo flow rehearsed
- Submission package complete

## Suggested Timeline (Tonight + Tomorrow)

### Tonight (2-3 hours)
1. hamsi: finalize model comparison and metrics
2. advaith: validate API endpoints and choose deployment path
3. you: tighten README and prepare demo script

### Tomorrow Morning (2 hours)
1. Merge all updates
2. Run one complete end-to-end test
3. Capture screenshots/videos of working outputs

### Tomorrow Final Hour
1. Freeze code
2. Final README proofread
3. Submission and backup upload

## Standardization Checklist (Do This Before Submission)
1. Keep file naming and section naming consistent
2. Do not leave commented dead code
3. Do not hardcode keys or secrets
4. Ensure requirements.txt includes all runtime deps
5. Ensure commands in README are copy-paste runnable
6. Keep logs/artifacts only if they support demo
7. Add concise failure learnings and future improvements

## Minimum Demo Script (Speak This)
1. "We optimized noisy social text before verification to reduce compute cost and increase throughput."
2. "We trained and compared three models: MLP, LinearSVC, RandomForest, then selected the best by test accuracy."
3. "Our API supports single, batch, and optional image-based inference."
4. "This architecture is deployable and scalable, with clear tradeoff between speed and accuracy."

## Command Block (Quick Copy)

1. Train and compare:
- .\\.venv\\Scripts\\python.exe compare_models.py --csv-root "News _dataset" --txt-root "FakeNewsData"

2. Run pipeline demo:
- .\\.venv\\Scripts\\python.exe main.py

3. Run API:
- .\\.venv\\Scripts\\python.exe -m uvicorn api:app --host 0.0.0.0 --port 8000

4. API test:
- curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "{\"text\":\"Breaking news: RBI is shutting all banks tomorrow\"}"

## Risk Plan (If Something Breaks)
1. If deployment fails, submit local demo video + endpoint logs + clear deploy steps
2. If OCR fails, continue with text path and mention OCR as optional extension
3. If Docker unavailable, use Railway Python runtime or local host demo
4. If model retraining is slow, use existing artifacts and rerun only inference

## Final Submission Checklist
1. README updated and clean
2. guide.md present and followed
3. API tested locally
4. Model artifacts generated
5. Screenshots/video captured
6. No leaked secrets
7. Team members know exact speaking parts
