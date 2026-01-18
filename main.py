import pandas as pd
import numpy as np
from sklearn import linear_model
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

# Pydantic model for input data
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

@app.post("/predict", response_model=List[PredictionResult])
def predict_student_counts(request: PredictionRequest):
    # Convert input data to DataFrame
    records = [{"Course": r.course, "Total_Students": r.total_students, "Year": r.year} for r in request.data]
    df = pd.DataFrame(records)
    
    data = df.sort_values(by=["Course", "Year"])
    courses = data["Course"].unique()
    
    predictions = []
    
    for course in courses:
        course_data = data[data["Course"] == course]
        X = course_data[["Year"]].values
        y = course_data["Total_Students"].values
        
        last_year = int(course_data["Year"].max())
        next_year = last_year + 1
        
        # Linear Regression Model
        model = linear_model.LinearRegression()
        model.fit(X, y)
        
        # Predict next year
        predicted_count = model.predict([[next_year]])[0]
        
        predictions.append(PredictionResult(
            course=course,
            predicted_year=next_year,
            predicted_count=int(round(predicted_count))
        ))
    
    return predictions