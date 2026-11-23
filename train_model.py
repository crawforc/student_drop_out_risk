# train_model.py - trains the dropout model, tests it, and saves it

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference, save_model

# ---- Step 1: Load the clean data ----
df = pd.read_csv("data/clean_student_data.csv")

# ---- Step 2: Split into training data (80%) and test data (20%) ----
train, test = train_test_split(
    df,
    test_size=0.20,            # 20% of students are kept aside for testing
    random_state=42,           # same split every time
    stratify=df["dropout"],    # keep the 32% dropout rate in both groups
)
print("Training students:", len(train))
print("Test students:", len(test))

# ---- Step 3: Prepare the data for the model ----
# Training data: the encoder LEARNS the categories
X_train, y_train, encoder = process_data(train, training=True)

# Test data: the encoder REUSES what it learned
X_test, y_test, _ = process_data(test, training=False, encoder=encoder)

# ---- Step 4: Train the model ----
model = train_model(X_train, y_train)
print("Model trained")

# ---- Step 5: Save the model and the encoder ----
save_model(model, "model/model.pkl")
save_model(encoder, "model/encoder.pkl")
print("Model and encoder saved in the model folder")

# ---- Step 6: Test the model on students it has never seen ----
preds = inference(model, X_test)
accuracy, precision, recall, f1 = compute_model_metrics(y_test, preds)

print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1:        {f1:.3f}")