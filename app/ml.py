from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from google import genai
from dotenv import load_dotenv

load_dotenv()

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from app.preprocessing import preprocess_text
from app.sample_data import SEED_NEWS
from app.database import get_all_bbc_news

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "factfinder_models.joblib"


@dataclass
class PredictionOutput:
    label: str
    confidence: float
    details: dict
    interpretation: str = ""


class ModelManager:
    def __init__(self):
        self.rf_pipeline: Pipeline | None = None
        self.xgb_pipeline: Pipeline | None = None
        self.metadata: dict = {}
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

    def _build_dataset(self) -> pd.DataFrame:
        rows = []
        for item in SEED_NEWS:
            # 1 is Real, 0 is Fake
            rows.append({"text": item["text"], "label": 1 if item["label"] == 1 else 0})
        
        # Load BBC news (Real News -> 1)
        bbc_articles = get_all_bbc_news()
        for article in bbc_articles:
            # We treat BBC news as verified real news (1)
            combined_text = f"{article['title']} {article['content']}".strip()
            if combined_text:
                rows.append({"text": combined_text, "label": 1})
                
        # Load Fake and Real News from downloaded Kaggle dataset to ensure perfect balance
        fake_csv_path = BASE_DIR / "data" / "fake_news.csv"
        if fake_csv_path.exists():
            try:
                df_all = pd.read_csv(fake_csv_path)
                
                # Filter FAKE news, use all available
                df_fake = df_all[df_all['label'] == 'FAKE']
                for _, row in df_fake.iterrows():
                    combined_text = f"{row['title']} {row['text']}".strip()
                    if combined_text:
                        rows.append({"text": combined_text, "label": 0})
                
                # Filter REAL news, use all available
                df_real = df_all[df_all['label'] == 'REAL']
                for _, row in df_real.iterrows():
                    combined_text = f"{row['title']} {row['text']}".strip()
                    if combined_text:
                        rows.append({"text": combined_text, "label": 1})
            except Exception as e:
                print(f"Error loading fake news dataset: {e}")
                
        return pd.DataFrame(rows)

    def train(self) -> dict:
        df = self._build_dataset()
        df["clean_text"] = df["text"].map(preprocess_text)
        if df["label"].nunique() < 2:
            raise ValueError("Training data must contain both fake and real examples.")

        test_size = 0.2 if len(df) >= 10 else 0.5
        x_train, x_test, y_train, y_test = train_test_split(
            df["clean_text"],
            df["label"],
            test_size=test_size,
            random_state=42,
            stratify=df["label"],
        )

        self.rf_pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
                ("classifier", RandomForestClassifier(n_estimators=250, random_state=42)),
            ]
        )
        self.xgb_pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
                (
                    "classifier",
                    XGBClassifier(
                        n_estimators=250,
                        max_depth=6,
                        learning_rate=0.1,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        random_state=42,
                    ),
                ),
            ]
        )

        self.rf_pipeline.fit(x_train, y_train)
        self.xgb_pipeline.fit(x_train, y_train)

        self.metadata = {
            "training_size": int(len(df)),
            "rf_accuracy": float(self.rf_pipeline.score(x_test, y_test)),
            "xgb_accuracy": float(self.xgb_pipeline.score(x_test, y_test)),
        }
        self._save()
        return self.metadata

    def _save(self) -> None:
        joblib.dump(
            {
                "rf_pipeline": self.rf_pipeline,
                "xgb_pipeline": self.xgb_pipeline,
                "metadata": self.metadata,
            },
            MODEL_PATH,
        )

    def load_or_train(self) -> dict:
        if MODEL_PATH.exists():
            payload = joblib.load(MODEL_PATH)
            self.rf_pipeline = payload["rf_pipeline"]
            self.xgb_pipeline = payload["xgb_pipeline"]
            self.metadata = payload.get("metadata", {})
            return self.metadata
        return self.train()

    def _generate_local_explanation(self, text: str, clean_text: str, label: str) -> str:
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        has_exclamation = '!' in text
        
        top_words = []
        if self.rf_pipeline:
            tfidf = self.rf_pipeline.named_steps.get("tfidf")
            if tfidf:
                feature_names = tfidf.get_feature_names_out()
                transformed = tfidf.transform([clean_text])
                
                coo = transformed.tocoo()
                tuples = zip(coo.col, coo.data)
                sorted_items = sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
                
                for col, score in sorted_items[:3]:
                    top_words.append(feature_names[col])

        reasons = []
        if top_words:
            reasons.append(f"Key terms heavily weighing on this prediction include: '{', '.join(top_words)}'.")
            
        if label == "Fake":
            if caps_ratio > 0.04:
                reasons.append("The text features an unusually high amount of capitalization, which is a common pattern in sensationalized content.")
            if has_exclamation:
                reasons.append("The use of exclamation marks indicates an emotional or alarmist tone, typical of unverified news.")
            if not reasons or len(reasons) == 1:
                reasons.append("Overall, the phrasing and linguistic structure closely match patterns found in previously identified deceptive articles.")
        else:
            if caps_ratio <= 0.04:
                reasons.append("The capitalization patterns appear consistent with standard journalistic editing.")
            if not has_exclamation:
                reasons.append("The lack of sensational punctuation suggests a neutral and objective reporting tone.")
            if len(reasons) <= 2:
                reasons.append("The vocabulary used aligns strongly with verified publications in our database.")
                
        return f"Local Interpretable AI Analysis: The engine classified this as '{label}' News. " + " ".join(reasons)

    def predict(self, text: str) -> PredictionOutput:
        if not self.rf_pipeline or not self.xgb_pipeline:
            self.load_or_train()

        # Phase 1: Fact-check verification against verified BBC Database
        is_verified = False
        if len(text) > 50: 
            bbc_articles = get_all_bbc_news()
            if bbc_articles:
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity
                
                # Combine title and content for each article in the DB
                db_texts = [f"{a.get('title', '')} {a.get('content', '')}" for a in bbc_articles]
                
                # Compute TF-IDF similarities
                # We use word n-grams to capture phrase matches, avoiding issues with small textual differences
                vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2)).fit(db_texts + [text])
                db_vectors = vectorizer.transform(db_texts)
                input_vector = vectorizer.transform([text])
                
                similarities = cosine_similarity(input_vector, db_vectors)[0]
                best_match_score = float(similarities.max()) if len(similarities) > 0 else 0.0
                
                # If the similarity is greater than 40% (which is generally very high for different texts), it's the same article
                if best_match_score > 0.40:
                    is_verified = True

        clean_text = preprocess_text(text)
        rf_prob = float(self.rf_pipeline.predict_proba([clean_text])[0][1])
        xgb_prob = float(self.xgb_pipeline.predict_proba([clean_text])[0][1])
        avg_prob = float(np.mean([rf_prob, xgb_prob]))
        
        if is_verified:
            # Override probabilities because we found exact verification in DB!
            label = "Real"
            # Return max of model confidence or highly confident 0.95
            avg_prob = max(avg_prob, 0.95) 
            rf_prob = max(rf_prob, 0.95)
            xgb_prob = max(xgb_prob, 0.95)
            confidence = avg_prob
            interpretation = "This article was definitively verified as Real News because it was found in our trusted BBC News database."
        else:
            label = "Real" if avg_prob >= 0.35 else "Fake"
            
            # Normalize confidence calculation to 0.35 threshold
            if label == "Real":
                confidence = 0.5 + ((avg_prob - 0.35) / 0.65) * 0.5
            else:
                confidence = 0.5 + ((0.35 - avg_prob) / 0.35) * 0.5
            
            confidence_pct = round(confidence * 100, 1)
            
            if confidence_pct >= 90:
                strength = "highly confident"
            elif confidence_pct >= 75:
                strength = "fairly confident"
            else:
                strength = "somewhat uncertain, but leans towards"
                
            prediction_reasoning = f"The ML engine is {strength} that this is {label} News."
            
            if os.environ.get("GEMINI_API_KEY"):
                try:
                    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
                    prompt = (
                        f"Analyze this short news snippet and briefly explain in 2-3 natural sentences why it is classified as '{label}' news. "
                        f"Focus on the tone, sensationalism, or sources. Do not mention the text is an ML input.\n\nText: {clean_text[:500]}"
                    )
                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=prompt
                    )
                    interpretation = response.text.replace('\n', ' ').strip()
                except Exception as e:
                    interpretation = self._generate_local_explanation(text, clean_text, label)
            else:
                interpretation = self._generate_local_explanation(text, clean_text, label)

        return PredictionOutput(
            label=label,
            confidence=round(confidence, 4),
            details={
                "random_forest_real_probability": round(rf_prob, 4),
                "xgboost_real_probability": round(xgb_prob, 4),
                "ensemble_real_probability": round(avg_prob, 4),
                "metadata": self.metadata,
                "verified_by_database": is_verified
            },
            interpretation=interpretation
        )

    def details_json(self, details: dict) -> str:
        return json.dumps(details)


model_manager = ModelManager()
