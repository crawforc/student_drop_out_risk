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

# Show all columns when printing
pd.set_option("display.max_columns", None)

# Summary stats for every number column (count, mean, min, max, etc.)
# .T flips the table so each column is a row - easier to read
# .round(2) keeps 2 decimal places
print(df.describe().T.round(2))

# What percent of students dropped out vs stayed?
print("Dropout rate:")
print(df["dropout"].value_counts(normalize=True).round(2))

# Compare the average of each number column for students who stayed (0) vs dropped out (1)
print("Averages by dropout:")
print(df.groupby("dropout").mean(numeric_only=True).T.round(2))
