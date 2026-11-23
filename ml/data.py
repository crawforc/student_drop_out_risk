# ml/data.py - turns the student data into numbers the model can read

import numpy as np
from sklearn.preprocessing import OneHotEncoder

# ---- Which columns the model uses ----

# Text columns: these need to be turned into numbers
CATEGORICAL_FEATURES = [
    "family_income_level",
    "course_type",
    "study_mode",
]

# Number columns: these can be used as they are
NUMERIC_FEATURES = [
    "age",
    "first_generation",
    "disability_status",
    "scholarship_holder",
    "tuition_up_to_date",
    "part_time_job",
    "weekly_work_hours",
    "semester",
    "international_student",
    "admission_grade",
    "previous_qualification_grade",
    "units_enrolled",
    "units_passed",
    "units_failed",
    "gpa_semester_1",
    "gpa_semester_2",
    "attendance_rate",
    "assignments_submitted_rate",
    "library_visits_per_month",
    "lms_logins_per_week",
    "extracurricular_activities",
    "missed_deadlines",
    "stress_level",
    "mental_health_support",
    "satisfaction_score",
    "commute_time_mins",
    "sleep_hours",
]

# NOTE: gender and nationality are NOT in either list above.
# They were left out on purpose so the model cannot use them (see model card).
# They stay in the data so we can still check fairness for each group.

# The column we want to predict
LABEL = "dropout"


def process_data(df, training=True, encoder=None):
    """
    Prepares the data for the model.

    df       - the table of student data
    training - True when training the model, False when making predictions
    encoder  - the saved encoder (only needed when training is False)

    Returns: X (the features), y (the answers), encoder
    """

    # Get the answers (dropout yes/no) if the column exists
    if LABEL in df.columns:
        y = df[LABEL].values
    else:
        y = np.array([])

    # Split the features into number columns and text columns
    X_numeric = df[NUMERIC_FEATURES].values
    X_categorical = df[CATEGORICAL_FEATURES].values

    # Turn text into numbers with one-hot encoding
    if training:
        # Learn the categories from the training data
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        X_categorical = encoder.fit_transform(X_categorical)
    else:
        # Use the categories the encoder already learned
        X_categorical = encoder.transform(X_categorical)

    # Join the number columns and the encoded text columns side by side
    X = np.concatenate([X_numeric, X_categorical], axis=1)

    return X, y, encoder