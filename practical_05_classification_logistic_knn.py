# Practical 5: Classification using Logistic Regression and K-Nearest Neighbors (KNN)
# Dataset: Admission_Predict.csv
# AIML Laboratory

import os
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

dataset_path = get_dataset_path("Admission_Predict.csv")
print(f"Loading dataset from: {dataset_path}")
df = pd.read_csv(dataset_path)

print("\n--- Dataset Head ---")
print(df.head())

print("\n--- Dataset Info ---")
df.info()

print(f"\nSize of dataset : {df.size}")
print(f"Shape of dataset: {df.shape}")

print("\n--- Statistical Summary ---")
print(df.describe())

print("\n--- Data Types ---")
print(df.dtypes.value_counts())

# Check for missing values
n = df.columns[df.dtypes != 'object']
print("\n--- Missing Values Percentage ---")
print(df[n].isnull().sum().sort_values(ascending=False) / len(df))

# Display Correlation Matrix
corr_matrix = df.corr(numeric_only=True)
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, linewidths=0.5, fmt=".2f", cmap="YlGnBu")
plt.title("Correlation Matrix - Admission Predict")
plt.tight_layout()
plt.show()

# Identify categorical and continuous columns
categorical_val = []
continous_val = []
for column in df.columns:
    if len(df[column].unique()) <= 10:
        categorical_val.append(column)
    else:
        continous_val.append(column)

if 'target' in categorical_val:
    categorical_val.remove('target')

dataset = pd.get_dummies(df, columns=categorical_val)

# Standardization of features
from sklearn.preprocessing import StandardScaler

s_sc = StandardScaler()
col_to_scale = [c for c in ['GRE Score', 'CGPA'] if c in dataset.columns]
if col_to_scale:
    dataset[col_to_scale] = s_sc.fit_transform(dataset[col_to_scale])

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")
    else:
        pred = clf.predict(X_test)
        clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        print("Test Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")


# Train-Test Split
from sklearn.model_selection import train_test_split

if 'target' not in dataset.columns:
    # If target not present (raw Kaggle chance of admit > 0.7), create binary target
    if 'Chance of Admit ' in dataset.columns:
        dataset['target'] = (dataset['Chance of Admit '] >= 0.7).astype(int)
    elif 'Chance of Admit' in dataset.columns:
        dataset['target'] = (dataset['Chance of Admit'] >= 0.7).astype(int)

X = dataset.drop('target', axis=1)
y = dataset['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Classification using Logistic Regression
print("\n" + "=" * 50)
print("LOGISTIC REGRESSION CLASSIFIER")
print("=" * 50)
from sklearn.linear_model import LogisticRegression

lr_clf = LogisticRegression(solver='liblinear')
lr_clf.fit(X_train, y_train)

print_score(lr_clf, X_train, y_train, X_test, y_test, train=True)
print_score(lr_clf, X_train, y_train, X_test, y_test, train=False)

lr_test_score = accuracy_score(y_test, lr_clf.predict(X_test)) * 100
lr_train_score = accuracy_score(y_train, lr_clf.predict(X_train)) * 100

results_df = pd.DataFrame(
    data=[["Logistic Regression", lr_train_score, lr_test_score]],
    columns=['Model', 'Training Accuracy %', 'Testing Accuracy %']
)

# 2. Classification using K-Nearest Neighbors (KNN)
print("\n" + "=" * 50)
print("K-NEAREST NEIGHBORS (KNN) CLASSIFIER")
print("=" * 50)
from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)

print_score(knn_clf, X_train, y_train, X_test, y_test, train=True)
print_score(knn_clf, X_train, y_train, X_test, y_test, train=False)

knn_test_score = accuracy_score(y_test, knn_clf.predict(X_test)) * 100
knn_train_score = accuracy_score(y_train, knn_clf.predict(X_train)) * 100

results_df_2 = pd.DataFrame(
    data=[["K-nearest neighbors", knn_train_score, knn_test_score]],
    columns=['Model', 'Training Accuracy %', 'Testing Accuracy %']
)

# Robust concat for all pandas versions
try:
    results_df = pd.concat([results_df, results_df_2], ignore_index=True)
except Exception:
    results_df = results_df.append(results_df_2, ignore_index=True)

print("\n" + "=" * 50)
print("FINAL MODEL COMPARISON SUMMARY")
print("=" * 50)
print(results_df.to_string(index=False))
