import pandas as pd
import numpy as np
from sklearn import linear_model
import pickle
import requests
import sys

# ── Configuration ──────────────────────────────────────────────
# Point this to your running enrollment system
API_URL = "http://localhost:3000/api/enrollment-forecast"

# ── Fetch data from the enrollment database via API ───────────
print("Fetching enrollment data from database...")
try:
    response = requests.get(API_URL, timeout=10)
    response.raise_for_status()
    records = response.json()
except requests.exceptions.ConnectionError:
    print(f"ERROR: Could not connect to {API_URL}")
    print("Make sure the enrollment app is running (npm run dev).")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)

if not records:
    print("No enrollment data found in the database.")
    sys.exit(1)

df = pd.DataFrame(records)
# Columns: course, year, total_students
df.columns = ["Course", "Year", "Total_Students"]
data = df.sort_values(by=["Course", "Year"])
courses = data["Course"].unique()

print(f"\nLoaded {len(data)} records for {len(courses)} courses:")
print(data.to_string(index=False))
print()

# Dictionary to store models for each course
models = {}

for course in courses:
    course_data = data[data["Course"] == course]
    X = course_data[["Year"]].values
    y = course_data["Total_Students"].values
    
    if len(X) < 2:
        print(f"Skipping {course} — needs at least 2 years of data (has {len(X)})")
        continue
    
    # Train Linear Regression Model
    model = linear_model.LinearRegression()
    model.fit(X, y)
    
    models[course] = model
    coef = model.coef_[0]
    intercept = model.intercept_
    print(f"Trained model for {course}  (slope={coef:.2f}, intercept={intercept:.2f})")

# Save all models to a pickle file
with open("models.pkl", "wb") as f:
    pickle.dump(models, f)

print("\nModels saved to models.pkl successfully!")
