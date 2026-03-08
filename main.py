import pandas as pd
import numpy as np
from sklearn import linear_model
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pickle
import requests
import os

app = FastAPI()

# Allow requests from the enrollment frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Configuration ──────────────────────────────────────────────
ENROLLMENT_API_URL = os.environ.get("ENROLLMENT_API_URL", "http://localhost:3000/api/enrollment-forecast")

# ── Pydantic Models ───────────────────────────────────────────
class StudentRecord(BaseModel):
    course: str
    total_students: int
    year: int

class PredictionRequest(BaseModel):
    data: List[StudentRecord]

class PredictionResult(BaseModel):
    course: str
    predicted_year: int
    predicted_count: int


def _run_prediction(df: pd.DataFrame, target_year: Optional[int] = None) -> List[PredictionResult]:
    """Shared logic: train a linear regression per course and return predictions."""
    data = df.sort_values(by=["Course", "Year"])
    courses = data["Course"].unique()
    predictions = []

    for course in courses:
        course_data = data[data["Course"] == course]
        X = course_data[["Year"]].values
        y = course_data["Total_Students"].values

        if len(X) < 2:
            continue

        last_year = int(course_data["Year"].max())
        predict_year = target_year if target_year else last_year + 1

        model = linear_model.LinearRegression()
        model.fit(X, y)

        predicted_count = model.predict([[predict_year]])[0]

        predictions.append(PredictionResult(
            course=course,
            predicted_year=predict_year,
            predicted_count=max(0, int(round(predicted_count)))
        ))

    return predictions


# ── POST /predict — accepts data from the enrollment app directly ──
@app.post("/predict", response_model=List[PredictionResult])
def predict_from_post(request: PredictionRequest, target_year: Optional[int] = Query(None)):
    """
    Accepts enrollment data via POST body and returns predictions.
    This is what the enrollment system calls.
    """
    records = [{"Course": r.course, "Total_Students": r.total_students, "Year": r.year} for r in request.data]
    if not records:
        return []
    df = pd.DataFrame(records)
    return _run_prediction(df, target_year)


# ── GET /predict — fetches data from the enrollment API ────────────
@app.get("/predict", response_model=List[PredictionResult])
def predict_from_get(target_year: Optional[int] = Query(None)):
    """
    Fetches live enrollment data from the enrollment-forecast API,
    trains a model per course, and returns predictions.
    """
    response = requests.get(ENROLLMENT_API_URL, timeout=10)
    response.raise_for_status()
    records = response.json()
    if not records:
        return []
    df = pd.DataFrame(records)
    df.columns = ["Course", "Year", "Total_Students"]
    return _run_prediction(df, target_year)


# ── GET /predict/pickle — uses pre-trained models ──────────────
@app.get("/predict/pickle", response_model=List[PredictionResult])
def predict_from_pickle(target_year: Optional[int] = Query(None)):
    """
    Uses pre-trained models from models.pkl for predictions.
    Run create_model.py first to generate the pickle file.
    """
    try:
        with open("models.pkl", "rb") as f:
            models = pickle.load(f)
    except FileNotFoundError:
        return []

    # We still need the data to know the last year per course
    response = requests.get(ENROLLMENT_API_URL, timeout=10)
    response.raise_for_status()
    records = response.json()
    df = pd.DataFrame(records)
    df.columns = ["Course", "Year", "Total_Students"]

    predictions = []

    for course, model in models.items():
        course_data = df[df["Course"] == course]
        last_year = int(course_data["Year"].max()) if len(course_data) > 0 else 2025
        predict_year = target_year if target_year else last_year + 1

        predicted_count = model.predict([[predict_year]])[0]

        predictions.append(PredictionResult(
            course=course,
            predicted_year=predict_year,
            predicted_count=max(0, int(round(predicted_count)))
        ))

    return predictions