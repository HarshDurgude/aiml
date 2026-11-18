# Practical 9: Model Optimization, Ensemble Learning & Hyperparameter Tuning (GridSearchCV)
# Models: SVM, Random Forest, AdaBoost
# Dataset: breast_cancer.csv
# AIML Laboratory

import os
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


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
    return None


# 1. Load Dataset
dataset_path = get_dataset_path("breast_cancer.csv")
if dataset_path and os.path.exists(dataset_path):
    print(f"Loading dataset from: {dataset_path}")
    data = pd.read_csv(dataset_path)
    X = data.drop(columns=["target"])
    y = data["target"]
else:
    print("Dataset file not found directly, loading from sklearn.datasets.load_breast_cancer...")
    raw_data = load_breast_cancer(as_frame=True)
    X = raw_data.data
    y = raw_data.target

print(f"Dataset Features Shape: {X.shape}")
print(f"Target Distribution:\n{y.value_counts()}")

# 2. Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Define Models and Parameter Grids for Hyperparameter Tuning
models = {
    "SVM": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC())
        ]),
        {
            "model__C": [0.1, 1, 10],
            "model__kernel": ["linear", "rbf"]
        }
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=42),
        {
            "n_estimators": [50, 100],
            "max_depth": [None, 5, 10]
        }
    ),
    "AdaBoost": (
        AdaBoostClassifier(random_state=42),
        {
            "n_estimators": [50, 100],
            "learning_rate": [0.5, 1.0]
        }
    )
}

results = []

print("\n" + "=" * 60)
print("TRAINING, EVALUATING AND OPTIMIZING ENSEMBLE & ML MODELS")
print("=" * 60)

# 4. Train, Evaluate and Optimize Each Model
for name, (model, params) in models.items():
    print(f"\n[+] Processing Model: {name} ...")

    # Evaluate Before Hyperparameter Optimization
    model.fit(X_train, y_train)
    pred_before = model.predict(X_test)
    before_accuracy = accuracy_score(y_test, pred_before)
    before_f1 = f1_score(y_test, pred_before)

    # Hyperparameter Tuning using 5-Fold Cross Validation
    grid = GridSearchCV(
        model,
        params,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )
    grid.fit(X_train, y_train)

    # Evaluate After Hyperparameter Optimization
    best_model = grid.best_estimator_
    pred_after = best_model.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy Before (%)": round(before_accuracy * 100, 2),
        "F1 Before": round(before_f1, 4),
        "Accuracy After (%)": round(accuracy_score(y_test, pred_after) * 100, 2),
        "Precision After": round(precision_score(y_test, pred_after), 4),
        "Recall After": round(recall_score(y_test, pred_after), 4),
        "F1 After": round(f1_score(y_test, pred_after), 4),
        "Best Parameters": str(grid.best_params_)
    })

# 5. Display Summary Results Table
print("\n" + "=" * 80)
print("HYPERPARAMETER TUNING & MODEL COMPARISON RESULTS")
print("=" * 80)
df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))
print("=" * 80)
