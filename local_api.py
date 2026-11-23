# local_api.py - sends requests to the running API and prints the results
# The API must be running first (uvicorn main:app --reload)

import requests

# The address of the API on this computer
url = "http://127.0.0.1:8000"

# ---- Test 1: GET the welcome message ----
r = requests.get(url)
print("GET status code:", r.status_code)
print("GET result:", r.json())

# ---- Test 2: POST a student and get a prediction ----
student = {
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

r = requests.post(url + "/predict", json=student)
print("POST status code:", r.status_code)
print("POST result:", r.json())