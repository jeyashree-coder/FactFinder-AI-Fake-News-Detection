from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, HttpUrl

from app.database import get_prediction_history, init_db, save_prediction, get_today_bbc_news
from app.ml import model_manager
from app.scraper import collect_bbc_news
from app.scheduler import start_scheduler

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"


class TextPredictionRequest(BaseModel):
    text: str


class UrlPredictionRequest(BaseModel):
    url: HttpUrl


def extract_text_from_url(url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
    }
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    title = soup.find("h1")
    paragraphs = [p.get_text(" ", strip=True) for p in soup.select("article p")]
    if not paragraphs:
        paragraphs = [p.get_text(" ", strip=True) for p in soup.select("p")]
    text = " ".join(paragraphs).strip()
    if title:
        text = f"{title.get_text(' ', strip=True)} {text}".strip()
    if len(text) < 80:
        raise ValueError("Unable to extract enough article text from the provided URL.")
    return text


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    start_scheduler()
    model_manager.load_or_train()
    yield


app = FastAPI(title="FactFinder", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def serve_index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/predict-text")
async def predict_text(payload: TextPredictionRequest):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text input cannot be empty.")

    result = model_manager.predict(payload.text)
    save_prediction(
        input_type="text",
        input_value=payload.text,
        prediction=result.label,
        confidence=result.confidence,
        model_details=model_manager.details_json(result.details),
    )
    return {"prediction": result.label, "confidence": result.confidence, "details": result.details, "interpretation": result.interpretation}


@app.post("/predict-url")
async def predict_url(payload: UrlPredictionRequest):
    try:
        extracted_text = extract_text_from_url(str(payload.url))
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Could not fetch the URL: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    result = model_manager.predict(extracted_text)

    # Domain Fact-Checking Override
    from urllib.parse import urlparse
    domain = urlparse(str(payload.url)).netloc.lower()
    if any(trusted in domain for trusted in ["bbc.com", "bbc.co.uk", "nytimes.com", "reuters.com", "apnews.com", "npr.org", "wsj.com"]):
        result.label = "Real"
        result.confidence = 0.98
        result.interpretation = f"This article and URL have been verified against our trusted publisher domain index ({domain}), virtually guaranteeing it is Real News."
        result.details["verified_by_database"] = True
    elif any(satire in domain for satire in ["theonion.com", "babylonbee.com", "clickhole.com"]):
        result.label = "Fake"
        result.confidence = 0.98
        result.interpretation = f"This article comes from {domain}, a known satirical or humor website, thus it is definitively classified as Fake News."
        result.details["verified_by_database"] = False

    save_prediction(
        input_type="url",
        input_value=str(payload.url),
        prediction=result.label,
        confidence=result.confidence,
        model_details=model_manager.details_json(result.details),
    )
    return {
        "prediction": result.label,
        "confidence": result.confidence,
        "details": result.details,
        "interpretation": result.interpretation,
        "extracted_preview": extracted_text[:500],
    }


@app.get("/history")
async def history():
    return {"history": get_prediction_history()}


@app.post("/collect-bbc-news")
async def trigger_collection():
    result = collect_bbc_news()
    if result.get("status") == "completed" and result.get("inserted", 0) > 0:
        model_manager.train()
    return result


@app.get("/today-news")
async def today_news():
    return {"news": get_today_bbc_news()}
