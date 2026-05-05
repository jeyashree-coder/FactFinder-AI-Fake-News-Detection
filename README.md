# FactFinder - Fake News Detection System

## Overview
AI-based web app to detect fake vs real news using ML and NLP.

## Features
- Text & URL input
- Fake/Real prediction with confidence
- Influential word highlighting (Explainable AI)
- BBC news scraping for validation

## Tech Stack
- Frontend: HTML, JS
- Backend: FastAPI
- ML: TF-IDF, XGBoost / Logistic Regression
- Database: SQLite

## Run Project
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload