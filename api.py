import os
import tempfile
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from main import (
    extract_text_from_image,
    extract_text_from_video,
    get_system_limitations,
    process_batch,
    process_post,
)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Input social/news text")


class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, description="List of input texts")


app = FastAPI(
    title="Automated Fact-Checker API",
    version="1.1.0",
    description="Inference API for optimized fake-news checking pipeline",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/limitations")
def limitations() -> dict:
    return {"system_limitations": get_system_limitations()}


@app.post("/predict")
def predict(request: PredictRequest) -> dict:
    return process_post(request.text)


@app.post("/predict-batch")
def predict_batch(request: BatchPredictRequest) -> dict:
    workers = min(8, max(1, len(request.texts)))
    results = process_batch(request.texts, workers=workers)
    return {"count": len(results), "results": results}


@app.post("/predict-image")
def predict_image(file: UploadFile = File(...)) -> dict:
    suffix = os.path.splitext(file.filename or "")[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        temp.write(file.file.read())
        temp_path = temp.name

    try:
        extracted = extract_text_from_image(temp_path)
        return process_post(extracted, is_image=False)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/predict-video")
def predict_video(file: UploadFile = File(...)) -> dict:
    suffix = os.path.splitext(file.filename or "")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        temp.write(file.file.read())
        temp_path = temp.name

    try:
        extracted = extract_text_from_video(temp_path)
        result = process_post(extracted, is_image=False)
        result["input_video"] = file.filename or temp_path
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
