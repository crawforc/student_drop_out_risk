# load pandas and data
import pandas as pd 
df = pd.read_csv("data/student_dropout_risk_2026.csv")

#exploratory data analysis
print("Rows and columns:", df.shape)
print(df.head())

#count missing values in each column
print("Missing values per column:")
print(df.isnull().sum())

#count rows that are copies
print("Duplicate rows:", df.duplicated().sum())

# Drop person_id column
df = df.drop(columns=["person_id"])

# List the text columns
text_columns = ["gender", "nationality", "family_income_level", "course_type", "study_mode"]

# Show values in text columns
for column in text_columns:
    print(df[column].value_counts())

# Save cleaned data as new file
df.to_csv("data/clean_student_data.csv", index=False)
print("Clean data saved. Rows and columns:", df.shape)