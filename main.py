import math
import os
import pickle
import re
from datetime import date
from typing import List, Optional

import pandas as pd
import requests
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn import linear_model

app = FastAPI()

# Allow requests from the enrollment frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
ENROLLMENT_API_URL = os.environ.get(
    "ENROLLMENT_API_URL",
    "http://localhost:3000/api/auth/student/forecast",
)
INCOMPLETE_YEAR_RATIO_THRESHOLD = float(
    os.environ.get("INCOMPLETE_YEAR_RATIO_THRESHOLD", "0.25")
)
INCOMPLETE_YEAR_MIN_DROP = int(os.environ.get("INCOMPLETE_YEAR_MIN_DROP", "15"))
MIN_HISTORY_FOR_INCOMPLETE_YEAR_CHECK = int(
    os.environ.get("MIN_HISTORY_FOR_INCOMPLETE_YEAR_CHECK", "4")
)
SCHOOL_YEAR_START_MONTH = int(os.environ.get("SCHOOL_YEAR_START_MONTH", "8"))
PREDICTION_SCHOOL_YEAR_OFFSET = int(
    os.environ.get("PREDICTION_SCHOOL_YEAR_OFFSET", "1")
)
SPARSE_HISTORY_BLEND_THRESHOLD = int(
    os.environ.get("SPARSE_HISTORY_BLEND_THRESHOLD", "3")
)
RECENT_VALUE_WEIGHT = float(os.environ.get("RECENT_VALUE_WEIGHT", "0.65"))
ANNUAL_MIN_RETENTION_RATIO = float(
    os.environ.get("ANNUAL_MIN_RETENTION_RATIO", "0.9")
)


# Pydantic Models - Enrollment
class StudentRecord(BaseModel):
    course: str
    total_students: int
    school_year: Optional[str] = None
    academic_year: Optional[str] = None
    year: Optional[int] = None


class PredictionResult(BaseModel):
    course: str
    predicted_school_year: str
    predicted_academic_year: str
    predicted_year: int
    predicted_count: int


# Pydantic Models - Capacity
class SectionHistoryRecord(BaseModel):
    """One row: how many sections existed for a program in a given school year."""

    program: str
    school_year: Optional[str] = None
    academic_year: Optional[str] = None
    year: Optional[int] = None
    student_count: int
    section_count: int
    avg_section_capacity: int


class RoomRecord(BaseModel):
    room_id: int
    capacity: int
    room_type: str
    status: str


class PredictionRequest(BaseModel):
    data: List[StudentRecord]
    section_history: Optional[List[SectionHistoryRecord]] = None
    rooms: Optional[List[RoomRecord]] = None
    target_year: Optional[int] = None
    target_school_year: Optional[str] = None


class CapacityRequest(BaseModel):
    """
    POST body for /predict/capacity.
    - section_history: historical rows (school_year/year, program, student_count, section_count, avg_section_capacity)
    - rooms: current room inventory from the DB
    - target_year: optional override for the prediction school year's start year
    """

    section_history: List[SectionHistoryRecord]
    rooms: List[RoomRecord]
    target_year: Optional[int] = None
    target_school_year: Optional[str] = None


class SectionRoomRecommendation(BaseModel):
    program: str
    predicted_school_year: str
    predicted_academic_year: str
    predicted_year: int
    predicted_students: int
    current_sections: int
    sections_needed: int
    sections_to_add: int
    add_section: bool
    additional_sections_needed: int
    recommended_sections: int
    avg_section_capacity: int
    current_capacity: int
    new_total_capacity: int
    utilization_rate: int
    total_capacity_needed: int
    available_room_slots: int
    rooms_to_add: int
    add_room: bool
    status: str
    recommendation: str


# Shared helpers
def _first_present(record: dict, *keys: str):
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def _school_year_label(start_year: int) -> str:
    return f"{start_year}-{start_year + 1}"


def _parse_school_year_value(
    school_year: Optional[str] = None,
    year: Optional[int] = None,
) -> int:
    if school_year:
        match = re.search(r"(?:19|20)\d{2}", str(school_year))
        if match:
            return int(match.group(0))
        raise ValueError(f"Invalid school_year format: {school_year}")

    if year is not None:
        return int(year)

    raise ValueError("A school_year or year value is required.")


def _resolve_target_school_year(
    target_year: Optional[int],
    target_school_year: Optional[str],
    default_start_year: int,
    today: Optional[date] = None,
) -> int:
    if target_school_year:
        return _parse_school_year_value(target_school_year, None)
    if target_year is not None:
        return int(target_year)

    default_prediction_start = _default_prediction_school_year_start(today)
    return max(default_start_year + 1, default_prediction_start)


def _current_school_year_start(today: Optional[date] = None) -> int:
    """
    Return the active school year's start year relative to the current date.
    With the default August start:
    - 2026-04-10 -> 2025
    - 2026-09-10 -> 2026
    """

    today = today or date.today()
    if today.month >= SCHOOL_YEAR_START_MONTH:
        return today.year
    return today.year - 1


def _default_prediction_school_year_start(today: Optional[date] = None) -> int:
    """
    Forecast ahead of the currently active school year.
    With the default offset of 1:
    - active 2025-2026 -> predict 2026-2027
    """

    return _current_school_year_start(today) + PREDICTION_SCHOOL_YEAR_OFFSET


def _trim_incomplete_latest_year(
    data: pd.DataFrame,
    *,
    year_column: str,
    value_column: str,
) -> pd.DataFrame:
    """
    Drop a clearly incomplete trailing year so partial current-school-year counts
    do not collapse the trend line.
    """

    ordered = data.sort_values(year_column)
    if len(ordered) < MIN_HISTORY_FOR_INCOMPLETE_YEAR_CHECK:
        return ordered

    latest_value = int(ordered[value_column].iloc[-1])
    previous_values = ordered[value_column].iloc[:-1].astype(float)
    baseline_value = float(previous_values.median())

    if baseline_value <= 0:
        return ordered

    if (
        latest_value <= (baseline_value * INCOMPLETE_YEAR_RATIO_THRESHOLD)
        and (baseline_value - latest_value) >= INCOMPLETE_YEAR_MIN_DROP
    ):
        return ordered.iloc[:-1].copy()

    return ordered


def _predict_enrollment_count(
    training_data: pd.DataFrame,
    *,
    year_column: str,
    value_column: str,
    predict_year: int,
) -> int:
    X = training_data[[year_column]].values
    y = training_data[value_column].astype(float).values

    if len(X) == 0:
        return 0

    if len(X) == 1:
        return max(0, int(round(y[-1])))

    last_observed_year = int(training_data[year_column].iloc[-1])
    last_observed_value = float(y[-1])
    forecast_horizon_years = max(1, predict_year - last_observed_year)

    model = linear_model.LinearRegression()
    model.fit(X, y)
    raw_prediction = float(model.predict([[predict_year]])[0])

    if len(training_data) <= SPARSE_HISTORY_BLEND_THRESHOLD:
        raw_prediction = (
            (last_observed_value * RECENT_VALUE_WEIGHT)
            + (raw_prediction * (1 - RECENT_VALUE_WEIGHT))
        )

    minimum_reasonable_prediction = last_observed_value * (
        ANNUAL_MIN_RETENTION_RATIO ** forecast_horizon_years
    )

    adjusted_prediction = max(raw_prediction, minimum_reasonable_prediction)
    return max(0, int(round(adjusted_prediction)))


def _normalize_enrollment_dataframe(records: List[dict]) -> pd.DataFrame:
    normalized = []

    for record in records:
        course = _first_present(record, "Course", "course")
        total_students = _first_present(record, "Total_Students", "total_students")
        school_year = _first_present(
            record,
            "School_Year",
            "school_year",
            "Academic_Year",
            "academic_year",
        )
        year = _first_present(record, "Year", "year")

        if course is None or total_students is None:
            raise ValueError("Each enrollment record must include course and total_students.")

        course = str(course).strip()
        school_year_value = _parse_school_year_value(school_year, year)
        normalized.append(
            {
                "Course": course,
                "School_Year": school_year or _school_year_label(school_year_value),
                "School_Year_Value": school_year_value,
                "Total_Students": int(total_students),
            }
        )

    return pd.DataFrame(normalized)


def _normalize_section_history_dataframe(
    section_history: List[SectionHistoryRecord],
) -> pd.DataFrame:
    normalized = []

    for record in section_history:
        item = record.dict() if hasattr(record, "dict") else dict(record)
        school_year = _first_present(
            item,
            "school_year",
            "School_Year",
            "academic_year",
            "Academic_Year",
        )
        year = _first_present(item, "year", "Year")
        school_year_value = _parse_school_year_value(school_year, year)
        program = str(item["program"]).strip()

        normalized.append(
            {
                **item,
                "program": program,
                "school_year": school_year or _school_year_label(school_year_value),
                "school_year_value": school_year_value,
            }
        )

    return pd.DataFrame(normalized)


def _run_prediction(
    df: pd.DataFrame,
    target_year: Optional[int] = None,
    target_school_year: Optional[str] = None,
) -> List[PredictionResult]:
    """Train a linear regression per course and return enrollment predictions by school year."""

    data = df.sort_values(by=["Course", "School_Year_Value"])
    courses = data["Course"].unique()
    predictions = []

    for course in courses:
        course_data = data[data["Course"] == course]
        training_data = _trim_incomplete_latest_year(
            course_data,
            year_column="School_Year_Value",
            value_column="Total_Students",
        )
        if len(training_data) < 2:
            continue

        last_year = int(course_data["School_Year_Value"].max())
        predict_year = _resolve_target_school_year(
            target_year,
            target_school_year,
            last_year,
        )

        predicted_count = _predict_enrollment_count(
            training_data,
            year_column="School_Year_Value",
            value_column="Total_Students",
            predict_year=predict_year,
        )
        predictions.append(
            PredictionResult(
                course=course,
                predicted_school_year=_school_year_label(predict_year),
                predicted_academic_year=_school_year_label(predict_year),
                predicted_year=predict_year,
                predicted_count=predicted_count,
            )
        )

    return predictions


def _run_capacity_prediction(
    section_history: List[SectionHistoryRecord],
    rooms: List[RoomRecord],
    target_year: Optional[int],
    target_school_year: Optional[str],
) -> List[SectionRoomRecommendation]:
    """
    For each program:
      1. Train LinearRegression: sections_needed = f(student_count)
      2. Predict enrollment for the target school year
      3. Predict sections needed for that enrollment
      4. Compare vs available room pool and surface recommendations
    """

    if not section_history:
        return []

    df = _normalize_section_history_dataframe(section_history)
    programs = df["program"].unique()

    available_rooms = [r for r in rooms if r.status.lower() == "available"]
    total_room_capacity = sum(r.capacity for r in available_rooms)
    allocated_capacity = 0

    results = []

    for program in programs:
        pdata = df[df["program"] == program].sort_values("school_year_value")
        training_data = _trim_incomplete_latest_year(
            pdata,
            year_column="school_year_value",
            value_column="student_count",
        )

        X_years = training_data[["school_year_value"]].values
        y_students = training_data["student_count"].values
        y_sections = training_data["section_count"].values
        avg_cap = int(pdata["avg_section_capacity"].median())

        last_year = int(pdata["school_year_value"].max())
        predict_year = _resolve_target_school_year(
            target_year,
            target_school_year,
            last_year,
        )

        if len(X_years) >= 2:
            predicted_students = _predict_enrollment_count(
                training_data,
                year_column="school_year_value",
                value_column="student_count",
                predict_year=predict_year,
            )
        else:
            predicted_students = int(y_students[-1])

        min_sections_by_capacity = (
            int(math.ceil(predicted_students / max(avg_cap, 1)))
            if predicted_students > 0
            else 0
        )

        if predicted_students <= avg_cap:
            sections_needed = min_sections_by_capacity
        elif len(y_students) >= 2:
            X_students = pdata[["student_count"]].values
            capacity_model = linear_model.LinearRegression()
            capacity_model.fit(X_students, y_sections)
            model_sections = max(
                0,
                int(math.ceil(capacity_model.predict([[predicted_students]])[0])),
            )
            sections_needed = max(min_sections_by_capacity, model_sections)
        else:
            sections_needed = min_sections_by_capacity

        current_sections = int(pdata["section_count"].iloc[-1])
        sections_to_add = max(0, sections_needed - current_sections)
        add_section = sections_to_add > 0
        additional_sections_needed = sections_to_add
        recommended_sections = sections_needed
        current_capacity = current_sections * avg_cap
        total_capacity_needed = sections_needed * avg_cap
        new_total_capacity = total_capacity_needed

        utilization_rate = (
            int(round((predicted_students / current_capacity) * 100))
            if current_capacity > 0
            else 0
        )

        remaining_room_capacity = total_room_capacity - allocated_capacity
        shortfall = total_capacity_needed - remaining_room_capacity
        rooms_to_add = (
            max(0, math.ceil(shortfall / max(avg_cap, 1)))
            if shortfall > 0
            else 0
        )
        add_room = rooms_to_add > 0
        allocated_capacity += total_capacity_needed

        lines = []
        if sections_to_add > 0:
            lines.append(
                f"Add {sections_to_add} section(s) - projected {predicted_students} students "
                f"needs {sections_needed} sections (currently {current_sections})."
            )
        else:
            lines.append(
                f"Current {current_sections} section(s) sufficient for projected "
                f"{predicted_students} students."
            )
        if rooms_to_add > 0:
            lines.append(
                f"Room capacity may be short by ~{max(0, shortfall)} seats - "
                f"consider adding {rooms_to_add} room(s)."
            )
        else:
            lines.append(
                "No additional room needed; available room pool is adequate for this program."
            )

        recommendation_text = " ".join(lines)

        results.append(
            SectionRoomRecommendation(
                program=program,
                predicted_school_year=_school_year_label(predict_year),
                predicted_academic_year=_school_year_label(predict_year),
                predicted_year=predict_year,
                predicted_students=predicted_students,
                current_sections=current_sections,
                sections_needed=sections_needed,
                sections_to_add=sections_to_add,
                add_section=add_section,
                additional_sections_needed=additional_sections_needed,
                recommended_sections=recommended_sections,
                avg_section_capacity=avg_cap,
                current_capacity=current_capacity,
                new_total_capacity=new_total_capacity,
                utilization_rate=utilization_rate,
                total_capacity_needed=total_capacity_needed,
                available_room_slots=total_room_capacity,
                rooms_to_add=rooms_to_add,
                add_room=add_room,
                status=recommendation_text,
                recommendation=recommendation_text,
            )
        )

    return results


@app.post("/predict")
def predict_from_post(request: PredictionRequest):
    """
    Single endpoint for all predictions.
    Always returns enrollment_predictions.
    If section_history and rooms are provided, also returns capacity_recommendations.
    """

    records = [r.dict() for r in request.data]
    if not records:
        return {"enrollment_predictions": [], "capacity_recommendations": []}

    df = _normalize_enrollment_dataframe(records)
    enrollment_predictions = _run_prediction(
        df,
        request.target_year,
        request.target_school_year,
    )

    capacity_recommendations = []
    if request.section_history and request.rooms:
        capacity_recommendations = _run_capacity_prediction(
            request.section_history,
            request.rooms,
            request.target_year,
            request.target_school_year,
        )

    return {
        "enrollment_predictions": [p.dict() for p in enrollment_predictions],
        "capacity_recommendations": [c.dict() for c in capacity_recommendations],
    }


@app.get("/predict", response_model=List[PredictionResult])
def predict_from_get(
    target_year: Optional[int] = Query(None),
    target_school_year: Optional[str] = Query(None),
):
    """Fetches live enrollment data and returns predictions by school year."""

    response = requests.get(ENROLLMENT_API_URL, timeout=10)
    response.raise_for_status()
    payload = response.json()
    records = payload.get("enrollment", payload) if isinstance(payload, dict) else payload
    if not records:
        return []

    df = _normalize_enrollment_dataframe(records)
    return _run_prediction(df, target_year, target_school_year)


@app.get("/predict/pickle", response_model=List[PredictionResult])
def predict_from_pickle(
    target_year: Optional[int] = Query(None),
    target_school_year: Optional[str] = Query(None),
):
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
    df = _normalize_enrollment_dataframe(records)

    predictions = []
    for course, model in models.items():
        normalized_course = str(course).strip()
        course_data = df[df["Course"] == normalized_course]
        last_year = int(course_data["School_Year_Value"].max()) if len(course_data) > 0 else 2025
        predict_year = _resolve_target_school_year(
            target_year,
            target_school_year,
            last_year,
        )
        predictions.append(
            PredictionResult(
                course=normalized_course,
                predicted_school_year=_school_year_label(predict_year),
                predicted_academic_year=_school_year_label(predict_year),
                predicted_year=predict_year,
                predicted_count=_predict_enrollment_count(
                    _trim_incomplete_latest_year(
                        course_data,
                        year_column="School_Year_Value",
                        value_column="Total_Students",
                    ),
                    year_column="School_Year_Value",
                    value_column="Total_Students",
                    predict_year=predict_year,
                ),
            )
        )

    return predictions
