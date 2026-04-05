import os
import pickle
import re
import sys

import pandas as pd
import requests
from sklearn import linear_model


API_URL = os.environ.get(
    "ENROLLMENT_API_URL",
    "http://localhost:3000/api/enrollment-forecast",
)


def _first_present(record: dict, *keys: str):
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def _school_year_label(start_year: int) -> str:
    return f"{start_year}-{start_year + 1}"


def _parse_school_year_value(school_year=None, year=None) -> int:
    if school_year:
        match = re.search(r"(?:19|20)\d{2}", str(school_year))
        if match:
            return int(match.group(0))
        raise ValueError(f"Invalid school_year format: {school_year}")

    if year is not None:
        return int(year)

    raise ValueError("A school_year or year value is required.")


print("Fetching enrollment data from database...")
try:
    response = requests.get(API_URL, timeout=10)
    response.raise_for_status()
    payload = response.json()
    records = payload.get("enrollment", payload) if isinstance(payload, dict) else payload
except requests.exceptions.ConnectionError:
    print(f"ERROR: Could not connect to {API_URL}")
    print("Make sure the enrollment app is running and the URL is correct.")
    sys.exit(1)
except Exception as exc:
    print(f"ERROR: {exc}")
    sys.exit(1)

if not records:
    print("No enrollment data found in the database.")
    sys.exit(1)

normalized = []
for record in records:
    course = _first_present(record, "Course", "course")
    total_students = _first_present(record, "Total_Students", "total_students")
    school_year = _first_present(record, "School_Year", "school_year")
    year = _first_present(record, "Year", "year")

    if course is None or total_students is None:
        print(f"Skipping invalid record: {record}")
        continue

    school_year_value = _parse_school_year_value(school_year, year)
    normalized.append(
        {
            "Course": course,
            "School_Year": school_year or _school_year_label(school_year_value),
            "School_Year_Value": school_year_value,
            "Total_Students": int(total_students),
        }
    )

if not normalized:
    print("No valid enrollment records were found.")
    sys.exit(1)

df = pd.DataFrame(normalized)
data = df.sort_values(by=["Course", "School_Year_Value"])
courses = data["Course"].unique()

print(f"\nLoaded {len(data)} records for {len(courses)} courses:")
print(data[["Course", "School_Year", "Total_Students"]].to_string(index=False))
print()

models = {}

for course in courses:
    course_data = data[data["Course"] == course]
    X = course_data[["School_Year_Value"]].values
    y = course_data["Total_Students"].values

    if len(X) < 2:
        print(
            f"Skipping {course} - needs at least 2 school years of data "
            f"(has {len(X)})"
        )
        continue

    model = linear_model.LinearRegression()
    model.fit(X, y)

    models[course] = model
    coef = model.coef_[0]
    intercept = model.intercept_
    print(f"Trained model for {course} (slope={coef:.2f}, intercept={intercept:.2f})")

with open("models.pkl", "wb") as f:
    pickle.dump(models, f)

print("\nModels saved to models.pkl successfully!")
