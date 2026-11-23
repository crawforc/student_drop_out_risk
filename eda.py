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

#---- EDA ----

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


# -----Charts----

import matplotlib.pyplot as plt

# One color for every chart
blue = "#2a78d6"

# Chart 1: How many students stayed vs dropped out?
counts = df["dropout"].value_counts().sort_index()
plt.figure(figsize=(6, 4))
plt.bar(["Stayed", "Dropped out"], counts, color=blue)
plt.title("Students who stayed vs dropped out")
plt.ylabel("Number of students")
plt.savefig("charts/01_dropout_counts.png", bbox_inches="tight")
plt.close()

# Chart 2: Semester 1 GPA for students who stayed vs dropped out
stayed_gpa = df[df["dropout"] == 0]["gpa_semester_1"]
dropped_gpa = df[df["dropout"] == 1]["gpa_semester_1"]
plt.figure(figsize=(6, 4))
plt.boxplot([stayed_gpa, dropped_gpa])
plt.xticks([1, 2], ["Stayed", "Dropped out"])
plt.title("Semester 1 GPA: stayed vs dropped out")
plt.ylabel("GPA")
plt.savefig("charts/02_gpa_by_dropout.png", bbox_inches="tight")
plt.close()

# Chart 3: Dropout rate for each family income level
income_rate = df.groupby("family_income_level")["dropout"].mean() * 100
income_rate = income_rate[["Low", "Middle", "High"]]
plt.figure(figsize=(6, 4))
plt.bar(income_rate.index, income_rate, color=blue)
plt.title("Dropout rate by family income level")
plt.ylabel("Dropout rate (%)")
plt.savefig("charts/03_dropout_by_income.png", bbox_inches="tight")
plt.close()

# Chart 4: How do columns correlate?
correlations = df.corr(numeric_only=True)["dropout"].drop("dropout").sort_values()
plt.figure(figsize=(7, 8))
plt.barh(correlations.index, correlations, color=blue)
plt.axvline(0, color="gray", linewidth=1)
plt.title("Correlation of each factor with dropout")
plt.xlabel("Correlation (negative = less likely to drop out)")
plt.savefig("charts/04_correlation_with_dropout.png", bbox_inches="tight")
plt.close()

print("4 charts saved in the charts folder")
