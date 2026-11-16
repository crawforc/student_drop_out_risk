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