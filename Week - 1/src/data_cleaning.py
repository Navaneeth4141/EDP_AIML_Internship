import pandas as pd

# Load the dataset
df = pd.read_csv("../dataset/student_data.csv")

# Display basic information
print("First 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Fill missing values with mean
df["Study_Hours"] = df["Study_Hours"].fillna(df["Study_Hours"].mean())
df["Assignment_Score"] = df["Assignment_Score"].fillna(df["Assignment_Score"].mean())

# Remove duplicate rows
df = df.drop_duplicates()

# Save cleaned dataset
df.to_csv("../cleaned_dataset/cleaned_student_data.csv", index=False)

print("\nData cleaned successfully!")
print("Cleaned dataset saved successfully.")