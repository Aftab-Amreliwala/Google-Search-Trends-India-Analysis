# Task 1 : Data Understanding

import pandas as pd

# Load the dataset using Pandas
df = pd.read_csv("google.csv")
print(df)

# Display

# First Five Rows
print("First 5 Rows")
print(df.head())

# Last Five Rows
print("Last 5 Rows")
print(df.tail())

# Print dataset shape
print("Dataset Shape:", df.shape)

# Print Columns Name
print("\nColumn Names:")
print(df.columns)

# Dataset size (total elements)
print("\nDataset Size:", df.size)

# Statistical Summary
print("\nStatistical Summary:")
print(df.describe())

# Identify

# Select quantitative (numerical) columns
quantitative_cols = df.select_dtypes(include=['int64', 'float64'])

print("Quantitative Columns:")
print(quantitative_cols.columns)

# Select qualitative (categorical) columns
qualitative_cols = df.select_dtypes(include=['object'])

print("Qualitative (Categorical) Columns:")
print(qualitative_cols.columns)

