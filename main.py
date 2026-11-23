# main.py - the FastAPI app that returns a dropout prediction for a student

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from ml.data import process_data
from ml.model import inference, load_model


# ---- Step 1: Describe what student data the API expects ----
class Student(BaseModel):
    age: int
    first_generation: int
    disability_status: int
    scholarship_holder: int
    tuition_up_to_date: int
    part_time_job: int
    weekly_work_hours: int
    semester: int
    international_student: int
    admission_grade: float
    previous_qualification_grade: float
    units_enrolled: int
    units_passed: int
    units_failed: int
    gpa_semester_1: float
    gpa_semester_2: float
    attendance_rate: float
    assignments_submitted_rate: float
    library_visits_per_month: int
    lms_logins_per_week: float
    extracurricular_activities: int
    missed_deadlines: int
    stress_level: int
    mental_health_support: int
    satisfaction_score: int
    commute_time_mins: int
    sleep_hours: float
    family_income_level: str
    course_type: str
    study_mode: str

    # An example student - this fills in the "Try it out" box on the docs page
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 25,
                    "first_generation": 1,
                    "disability_status": 1,
                    "scholarship_holder": 1,
                    "tuition_up_to_date": 0,
                    "part_time_job": 1,
                    "weekly_work_hours": 12,
                    "semester": 8,
                    "international_student": 0,
                    "admission_grade": 119.3,
                    "previous_qualification_grade": 195.6,
                    "units_enrolled": 13,
                    "units_passed": 10,
                    "units_failed": 3,
                    "gpa_semester_1": 2.33,
                    "gpa_semester_2": 1.42,
                    "attendance_rate": 81.9,
                    "assignments_submitted_rate": 97.9,
                    "library_visits_per_month": 2,
                    "lms_logins_per_week": 3.3,
                    "extracurricular_activities": 1,
                    "missed_deadlines": 2,
                    "stress_level": 7,
                    "mental_health_support": 1,
                    "satisfaction_score": 6,
                    "commute_time_mins": 19,
                    "sleep_hours": 6.1,
                    "family_income_level": "Low",
                    "course_type": "STEM",
                    "study_mode": "Full-time",
                }
            ]
        }
    }


# ---- Step 2: Load the saved model and encoder (once, when the app starts) ----
model = load_model("model/model.pkl")
encoder = load_model("model/encoder.pkl")

# ---- Step 3: Create the app ----
app = FastAPI(title="Student Dropout Risk API")


# ---- Step 4: GET - a welcome message to show the API is running ----
@app.get("/")
async def welcome():
    return {"message": "Welcome to the Student Dropout Risk API"}


# ---- Step 5: POST - send a student's data, get a prediction back ----
@app.post("/predict")
async def predict(student: Student):
    # Turn the student's data into a one-row table
    df = pd.DataFrame([student.model_dump()])

    # Prepare it the same way as the training data
    X, _, _ = process_data(df, training=False, encoder=encoder)

    # Make the prediction (1 = at risk, 0 = not at risk)
    pred = inference(model, X)[0]

    # Also get the model's confidence that the student will drop out
    risk_score = model.predict_proba(X)[0][1]

    if pred == 1:
        result = "At risk of dropping out"
    else:
        result = "Not at risk"

    return {"prediction": result, "risk_score": round(float(risk_score), 2)}