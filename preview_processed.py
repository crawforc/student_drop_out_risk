# preview_processed.py - saves a small sample of what the model sees, so we can look at it

import pandas as pd
from ml.data import process_data, NUMERIC_FEATURES, CATEGORICAL_FEATURES

# Load the clean data and process it
df = pd.read_csv("data/clean_student_data.csv")
X, y, encoder = process_data(df)

# Build readable names for the new one-hot columns (e.g. study_mode_Online)
encoded_names = list(encoder.get_feature_names_out(CATEGORICAL_FEATURES))
all_names = NUMERIC_FEATURES + encoded_names

# Turn the numbers back into a table with column names, and keep the first 20 rows
preview = pd.DataFrame(X, columns=all_names).head(20)
preview["dropout"] = y[:20]

# Save it so we can open it in Data Wrangler
preview.to_csv("data/processed_preview.csv", index=False)
print("Saved data/processed_preview.csv with", len(all_names), "feature columns")
print(encoded_names)