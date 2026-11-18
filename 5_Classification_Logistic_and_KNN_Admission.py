# Practical 5: Classification using Logistic Regression & KNN
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# 1. Load Dataset
path = "Admission_Predict.csv"
df = pd.read_csv(path)

# 2. Target Definition & Feature Selection
if 'target' not in df.columns:
    chance_col = 'Chance of Admit ' if 'Chance of Admit ' in df.columns else 'Chance of Admit'
    df['target'] = (df[chance_col] >= 0.7).astype(int)

features = [c for c in ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research'] if c in df.columns]
X = df[features]
y = df['target']

# 3. Standardization & Train-Test Split (80:20)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, lr_pred) * 100:.2f}%")
print("Confusion Matrix:\n", confusion_matrix(y_test, lr_pred))

# 5. K-Nearest Neighbors (KNN)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
print(f"\nKNN Accuracy: {accuracy_score(y_test, knn_pred) * 100:.2f}%")
print("KNN Classification Report:\n", classification_report(y_test, knn_pred))
