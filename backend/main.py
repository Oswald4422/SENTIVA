import re
import warnings
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DATASET_PATH = BASE_DIR / "Dataset" / "final_all_multilingual_tweets(1).csv"
FRONTEND_DIR = BASE_DIR / "frontend"
MODEL_PATH = BASE_DIR / "backend" / "sentiva_model.pkl"

# ── Preprocessing ──────────────────────────────────────────────────────────────
NORM_MAP = {
    "gyimii": "gyimi",
    "agyimi": "gyimi",
    "nkwasia": "fool",
    "kwasea": "fool",
    "paa": "very",
    "ɛyɛ": "eye",
    "ɔ": "o",
    "ɛ": "e",
}


def normalize_local(text: str) -> str:
    for k, v in NORM_MAP.items():
        text = text.replace(k, v)
    return text


def preprocess(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return normalize_local(text)


# ── Model ──────────────────────────────────────────────────────────────────────
LABEL_NAMES = ["Negative", "Neutral", "Positive"]
le = LabelEncoder()
pipeline: Optional[Pipeline] = None


def train():
    global pipeline, le

    if MODEL_PATH.exists():
        print("[SENTIVA] Loading model from cache...")
        pipeline, le = joblib.load(MODEL_PATH)
        print("[SENTIVA] Model loaded from cache.")
        return

    print("[SENTIVA] Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    df.columns = df.columns.str.strip().str.lower()
    df = df.dropna(subset=["tweets", "labels"])
    df["clean"] = df["tweets"].apply(preprocess)

    df["labels"] = df["labels"].str.strip().str.capitalize()
    label_map = {
        "Neutral-negative": "Neutral",
        "Neutral-positive": "Neutral",
        "Positive-neutral": "Positive",
        "Negative-neutral": "Negative",
    }
    df["labels"] = df["labels"].replace(label_map)
    df = df[df["labels"].isin(LABEL_NAMES)]

    y = le.fit_transform(df["labels"])
    X = df["clean"]

    print(f"[SENTIVA] Training on {len(X)} samples | Classes: {le.classes_.tolist()}")

    tfidf = TfidfVectorizer(
        max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.95
    )

    base_estimators = [
        ("nb", MultinomialNB()),
        ("lr", LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
        ("svc", CalibratedClassifierCV(LinearSVC(max_iter=2000, random_state=42))),
    ]

    stacking = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=5,
        passthrough=False,
        n_jobs=-1,
    )

    pipeline = Pipeline([("tfidf", tfidf), ("clf", stacking)])
    pipeline.fit(X, y)
    joblib.dump((pipeline, le), MODEL_PATH)
    print("[SENTIVA] Model trained and saved to cache.")


train()

# ── FastAPI ────────────────────────────────────────────────────────────────────
app = FastAPI(title="SENTIVA API", docs_url="/api/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextIn(BaseModel):
    text: str


def _predict_one(text: str) -> dict:
    clean = preprocess(text)
    proba = pipeline.predict_proba([clean])[0]
    idx = int(np.argmax(proba))
    classes = le.classes_.tolist()
    return {
        "label": classes[idx],
        "confidence": float(proba[idx]),
        "scores": {c: float(p) for c, p in zip(classes, proba)},
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "classes": le.classes_.tolist()}


@app.post("/api/predict")
def predict(body: TextIn):
    if not body.text.strip():
        raise HTTPException(400, "Text cannot be empty.")
    return _predict_one(body.text)


@app.post("/api/predict-file")
async def predict_file(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".txt"):
        raise HTTPException(400, "Only .txt files are supported.")

    raw = await file.read()
    text = raw.decode("utf-8", errors="ignore")
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    if not lines:
        raise HTTPException(400, "The uploaded file contains no text.")

    lines = lines[:200]
    results = []
    summary = {"Negative": 0, "Neutral": 0, "Positive": 0}

    for line in lines:
        r = _predict_one(line)
        results.append({"text": line, **r})
        summary[r["label"]] += 1

    dominant = max(summary, key=summary.get)
    return {
        "overall_label": dominant,
        "summary": summary,
        "total_analyzed": len(results),
        "results": results,
    }


# Serve frontend — must be registered last
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="static")
