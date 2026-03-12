import pandas as pd
import numpy as np
from sklearn import linear_model
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import math
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
ENROLLMENT_API_URL = os.environ.get("ENROLLMENT_API_URL", "http://localhost:3000/api/auth/student/forecast")

# ── Pydantic Models — Enrollment ──────────────────────────────
class StudentRecord(BaseModel):
    course: str
    total_students: int
    year: int

class PredictionResult(BaseModel):
    course: str
    predicted_year: int
    predicted_count: int


# ── Pydantic Models — Capacity ────────────────────────────────
class SectionHistoryRecord(BaseModel):
    """One row: how many sections existed for a program in a given year."""
    program: str
    year: int
    student_count: int       # total enrolled that year
    section_count: int       # how many sections existed
    avg_section_capacity: int  # max_capacity of a typical section

class RoomRecord(BaseModel):
    room_id: int
    capacity: int
    room_type: str           # e.g. "lecture", "lab"
    status: str              # "available" | "occupied" | "maintenance"

class PredictionRequest(BaseModel):
    data: List[StudentRecord]
    section_history: Optional[List[SectionHistoryRecord]] = None
    rooms: Optional[List[RoomRecord]] = None
    target_year: Optional[int] = None

class CapacityRequest(BaseModel):
    """
    POST body for /predict/capacity.
    - section_history: historical rows (year, program, student_count, section_count, avg_section_capacity)
    - rooms: current room inventory from the DB
    - target_year: optional override for the prediction year
    """
    section_history: List[SectionHistoryRecord]
    rooms: List[RoomRecord]
    target_year: Optional[int] = None

class SectionRoomRecommendation(BaseModel):
    program: str
    predicted_year: int
    predicted_students: int
    # Section predictions
    current_sections: int          # latest known section count
    sections_needed: int           # model prediction
    sections_to_add: int           # max(0, needed - current)
    avg_section_capacity: int
    # Room analysis (shared pool)
    total_capacity_needed: int     # sections_needed * avg_section_capacity
    available_room_slots: int      # sum of capacities of "available" rooms
    rooms_to_add: int              # extra rooms required if pool is short
    recommendation: str            # human-readable advice


# ── Shared helpers ─────────────────────────────────────────────
def _run_prediction(df: pd.DataFrame, target_year: Optional[int] = None) -> List[PredictionResult]:
    """Train a linear regression per course and return enrollment predictions."""
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


def _run_capacity_prediction(
    section_history: List[SectionHistoryRecord],
    rooms: List[RoomRecord],
    target_year: Optional[int],
) -> List[SectionRoomRecommendation]:
    """
    For each program:
      1. Train LinearRegression: sections_needed = f(student_count)
      2. Predict enrollment for target_year (also via LinearRegression on students)
      3. Predict sections needed for that enrollment
      4. Compare vs available room pool and surface recommendations
    """
    if not section_history:
        return []

    df = pd.DataFrame([r.dict() for r in section_history])
    programs = df["program"].unique()

    # Total available room capacity (ignore occupied / maintenance)
    available_rooms = [r for r in rooms if r.status.lower() == "available"]
    total_room_capacity = sum(r.capacity for r in available_rooms)
    # Track how much room capacity has been "allocated" as we iterate programs
    allocated_capacity = 0

    results = []

    for program in programs:
        pdata = df[df["program"] == program].sort_values("year")

        X_years = pdata[["year"]].values
        y_students = pdata["student_count"].values
        y_sections = pdata["section_count"].values
        avg_cap = int(pdata["avg_section_capacity"].median())

        last_year = int(pdata["year"].max())
        predict_year = target_year if target_year else last_year + 1

        # -- Model A: predict enrollment for target year
        if len(X_years) >= 2:
            enroll_model = linear_model.LinearRegression()
            enroll_model.fit(X_years, y_students)
            predicted_students = max(0, int(round(enroll_model.predict([[predict_year]])[0])))
        else:
            predicted_students = int(y_students[-1])

        # -- Model B: sections needed = f(student_count)
        if len(y_students) >= 2:
            X_students = pdata[["student_count"]].values
            capacity_model = linear_model.LinearRegression()
            capacity_model.fit(X_students, y_sections)
            sections_needed = max(1, int(math.ceil(
                capacity_model.predict([[predicted_students]])[0]
            )))
        else:
            # Fallback: simple ceiling division
            sections_needed = max(1, math.ceil(predicted_students / max(avg_cap, 1)))

        current_sections = int(pdata["section_count"].iloc[-1])
        sections_to_add = max(0, sections_needed - current_sections)
        total_capacity_needed = sections_needed * avg_cap

        # Room allocation from shared pool
        remaining_room_capacity = total_room_capacity - allocated_capacity
        shortfall = total_capacity_needed - remaining_room_capacity
        rooms_to_add = max(0, math.ceil(shortfall / max(avg_cap, 1))) if shortfall > 0 else 0
        allocated_capacity += total_capacity_needed

        # Build recommendation text
        lines = []
        if sections_to_add > 0:
            lines.append(
                f"Add {sections_to_add} section(s) — projected {predicted_students} students "
                f"needs {sections_needed} sections (currently {current_sections})."
            )
        else:
            lines.append(
                f"Current {current_sections} section(s) sufficient for projected "
                f"{predicted_students} students."
            )
        if rooms_to_add > 0:
            lines.append(
                f"Room capacity may be short by ~{max(0,shortfall)} seats — "
                f"consider adding {rooms_to_add} room(s)."
            )
        else:
            lines.append("Available room pool is adequate for this program.")

        results.append(SectionRoomRecommendation(
            program=program,
            predicted_year=predict_year,
            predicted_students=predicted_students,
            current_sections=current_sections,
            sections_needed=sections_needed,
            sections_to_add=sections_to_add,
            avg_section_capacity=avg_cap,
            total_capacity_needed=total_capacity_needed,
            available_room_slots=total_room_capacity,
            rooms_to_add=rooms_to_add,
            recommendation=" ".join(lines),
        ))

    return results


# ── POST /predict — enrollment + capacity predictions ────────
@app.post("/predict")
def predict_from_post(request: PredictionRequest):
    """
    Single endpoint for all predictions.
    Always returns enrollment_predictions.
    If section_history and rooms are provided, also returns capacity_recommendations.
    """
    records = [{"Course": r.course, "Total_Students": r.total_students, "Year": r.year} for r in request.data]
    if not records:
        return {"enrollment_predictions": [], "capacity_recommendations": []}

    df = pd.DataFrame(records)
    enrollment_predictions = _run_prediction(df, request.target_year)

    capacity_recommendations = []
    if request.section_history and request.rooms:
        capacity_recommendations = _run_capacity_prediction(
            request.section_history,
            request.rooms,
            request.target_year,
        )

    return {
        "enrollment_predictions": [p.dict() for p in enrollment_predictions],
        "capacity_recommendations": [c.dict() for c in capacity_recommendations],
    }


# ── GET /predict — enrollment predictions (live data) ────────
@app.get("/predict", response_model=List[PredictionResult])
def predict_from_get(target_year: Optional[int] = Query(None)):
    """Fetches live enrollment data from the enrollment API and returns predictions."""
    response = requests.get(ENROLLMENT_API_URL, timeout=10)
    response.raise_for_status()
    payload = response.json()
    # Handle both old flat-array format and new object format { enrollment: [...], ... }
    records = payload.get("enrollment", payload) if isinstance(payload, dict) else payload
    if not records:
        return []
    df = pd.DataFrame(records)
    df = df.rename(columns={"course": "Course", "year": "Year", "total_students": "Total_Students"})
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

    response = requests.get(ENROLLMENT_API_URL, timeout=10)
    response.raise_for_status()
    payload = response.json()
    records = payload.get("enrollment", payload) if isinstance(payload, dict) else payload
    df = pd.DataFrame(records)
    df = df.rename(columns={"course": "Course", "year": "Year", "total_students": "Total_Students"})

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