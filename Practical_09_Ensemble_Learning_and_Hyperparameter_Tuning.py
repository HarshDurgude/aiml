# Practical 9: Ensemble Learning & Hyperparameter Tuning (GridSearchCV)
import os
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC

# 1. Load Dataset
path = "/home/avcoe/breast_cancer.csv" if os.path.exists("/home/avcoe/breast_cancer.csv") else "datasets/breast_cancer.csv"
if os.path.exists(path):
    df = pd.read_csv(path)
    X, y = df.drop(columns=["target"]), df["target"]
else:
    raw = load_breast_cancer(as_frame=True)
    X, y = raw.data, raw.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Define Models and Parameter Grids
models = {
    "SVM": (SVC(), {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}),
    "Random Forest": (RandomForestClassifier(random_state=42), {"n_estimators": [50, 100]}),
    "AdaBoost": (AdaBoostClassifier(random_state=42), {"n_estimators": [50, 100], "learning_rate": [0.5, 1.0]})
}

# 3. Train, Optimize & Compare
print("--- MODEL OPTIMIZATION & HYPERPARAMETER TUNING ---")
for name, (model, params) in models.items():
    # Before Tuning
    model.fit(X_train, y_train)
    acc_before = accuracy_score(y_test, model.predict(X_test))

    # After Tuning with 5-Fold Cross Validation
    grid = GridSearchCV(model, params, cv=5, scoring="accuracy")
    grid.fit(X_train, y_train)
    acc_after = accuracy_score(y_test, grid.best_estimator_.predict(X_test))

    print(f"\n{name}:")
    print(f"  Accuracy Before: {acc_before * 100:.2f}% | After Tuning: {acc_after * 100:.2f}%")
    print(f"  Best Parameters: {grid.best_params_}")
