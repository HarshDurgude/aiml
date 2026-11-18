# Practical 4: Data Preprocessing, Exploratory Data Analysis (EDA) and Train-Test Split
# Dataset: Heart Disease Dataset (heart.csv)
# AIML Laboratory

import os
import statistics as s
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def get_dataset_path(filename):
    """Find dataset either in /home/avcoe/ (college PC) or local datasets/ folder (laptop)."""
    candidates = [
        f"/home/avcoe/{filename}",
        os.path.join(os.path.dirname(__file__), "datasets", filename),
        os.path.join("datasets", filename),
        filename
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return f"/home/avcoe/{filename}"


sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")

dataset_path = get_dataset_path("heart.csv")
print(f"Loading dataset from: {dataset_path}")
df = pd.read_csv(dataset_path)

print("\n--- First 5 Rows ---")
print(df.head())

# Information of the dataset
print("\n--- Dataset Info ---")
df.info()

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nData Type Counts:")
print(df.dtypes.value_counts())

print("\n--- Statistical Summary ---")
print(df.describe())

# Dataframe size and shape
print("\nSize of dataset is :", df.size)
print("Shape of dataset is :", df.shape)

# Missing values check
n = df.columns[df.dtypes != 'object']
print("\nMissing values in numerical columns:")
print(df[n].isnull().sum().sort_values(ascending=False))

# Fill missing Age values with median (if any)
if 'age' in df.columns:
    df['age'] = df['age'].fillna(df['age'].median())
    print("\nMissing Values After Handling:")
    print(df.isnull().sum())

# Remove duplicate records
df = df.drop_duplicates()
print("\nShape after removing duplicates:", df.shape)

# --------------------------------------------------
# Statistical Analysis
# --------------------------------------------------
print("\n--- Mean (Numeric Only) ---")
print(df.mean(numeric_only=True))

print("\n--- Median (Numeric Only) ---")
print(df.median(numeric_only=True))

print("\n--- Standard Deviation (Numeric Only) ---")
print(df.std(numeric_only=True))

print("\n--- Correlation Matrix ---")
print(df.corr(numeric_only=True))

# --------------------------------------------------
# Data Visualization (Non-blocking or interactive)
# --------------------------------------------------
# Histogram of Age
if 'age' in df.columns:
    plt.figure(figsize=(8, 5))
    plt.hist(df['age'], bins=20, edgecolor='black')
    plt.title('Histogram of Age')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

# Histogram of Cholesterol
if 'chol' in df.columns:
    plt.figure(figsize=(8, 5))
    plt.hist(df['chol'], bins=15, edgecolor='black')
    plt.title('Histogram of Cholesterol')
    plt.xlabel('Cholesterol')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

# Boxplot - Age
if 'age' in df.columns:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df['age'])
    plt.title('Boxplot of Age')
    plt.tight_layout()
    plt.show()

# Boxplot - Cholesterol
if 'chol' in df.columns:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df['chol'])
    plt.title('Boxplot of Cholesterol')
    plt.tight_layout()
    plt.show()

# Finding mean age of Patients
if 'age' in df.columns:
    average = s.mean(df['age'])
    print("\nAverage age of patients: ", average)

# --------------------------------------------------
# Train-Test Split
# --------------------------------------------------
from sklearn.model_selection import train_test_split

if 'target' in df.columns:
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=56)

    print("\n--- Split Shapes ---")
    print(f"X_train Shape: {X_train.shape}")
    print(f"X_test Shape : {X_test.shape}")
    print(f"y_train Shape: {y_train.shape}")
    print(f"y_test Shape : {y_test.shape}")
    print("\nPractical 4 execution completed successfully!")
