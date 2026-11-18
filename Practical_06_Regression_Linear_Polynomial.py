# Practical 6: Regression Analysis (Linear Regression and Polynomial Regression)
# Dataset: Housing.csv
# AIML Laboratory

import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


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


# Load dataset
dataset_path = get_dataset_path("Housing.csv")
print(f"Loading dataset from: {dataset_path}")
data = pd.read_csv(dataset_path)

print("Dataset loaded successfully.")

# Display dataset information
print("\nDataset shape:")
print(data.shape)

print("\nFirst 5 rows:")
print(data.head())

print("\nColumn names:")
print(data.columns.tolist())

print("\nData types:")
print(data.dtypes)

print("\nMissing values in original dataset:")
print(data.isnull().sum())

# Define target column
TARGET_COLUMN = "price"

if TARGET_COLUMN not in data.columns:
    raise ValueError(
        f"Target column '{TARGET_COLUMN}' not found.\n"
        f"Available columns are:\n"
        f"{data.columns.tolist()}"
    )

# Separate input features and target
X = data.drop(columns=[TARGET_COLUMN])
y = data[TARGET_COLUMN]

print("\nInput shape :", X.shape)
print("Target shape:", y.shape)

# Identify numerical and categorical features
numeric_columns = X.select_dtypes(include=["number"]).columns
categorical_columns = X.select_dtypes(include=["object", "category"]).columns

print("\nNumerical columns  :", numeric_columns.tolist())
print("Categorical columns:", categorical_columns.tolist())

# Handle missing numerical values
X[numeric_columns] = X[numeric_columns].fillna(X[numeric_columns].median())

# Handle missing categorical values
for column in categorical_columns:
    if X[column].isnull().any():
        X[column] = X[column].fillna(X[column].mode()[0])

# Convert categorical columns to numeric
X = pd.get_dummies(X, columns=categorical_columns, drop_first=False, dtype=int)

print("\nColumns after categorical encoding:")
print(X.columns.tolist())

# Check for missing values
print("\nTotal missing input values:", X.isnull().sum().sum())
print("Missing target values:", y.isnull().sum())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))
print("Number of input features after encoding:", X.shape[1])

# --------------------------------------------------
# 1. Linear Regression
# --------------------------------------------------
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

linear_predictions = linear_model.predict(X_test)

linear_mse = mean_squared_error(y_test, linear_predictions)
linear_mae = mean_absolute_error(y_test, linear_predictions)
linear_r2 = r2_score(y_test, linear_predictions)

print("\n" + "=" * 50)
print("LINEAR REGRESSION METRICS")
print("=" * 50)
print(f"Mean Squared Error (MSE) : {linear_mse:.6f}")
print(f"Mean Absolute Error (MAE): {linear_mae:.6f}")
print(f"R² Score                 : {linear_r2:.6f}")

# --------------------------------------------------
# 2. Polynomial Regression (Degree = 2)
# --------------------------------------------------
polynomial_model = Pipeline([
    ("polynomial_features", PolynomialFeatures(degree=2, include_bias=False)),
    ("scaler", StandardScaler()),
    ("linear_regression", LinearRegression())
])

polynomial_model.fit(X_train, y_train)
polynomial_predictions = polynomial_model.predict(X_test)

polynomial_mse = mean_squared_error(y_test, polynomial_predictions)
polynomial_mae = mean_absolute_error(y_test, polynomial_predictions)
polynomial_r2 = r2_score(y_test, polynomial_predictions)

print("\n" + "=" * 50)
print("POLYNOMIAL REGRESSION METRICS (Degree 2)")
print("=" * 50)
print(f"Mean Squared Error (MSE) : {polynomial_mse:.6f}")
print(f"Mean Absolute Error (MAE): {polynomial_mae:.6f}")
print(f"R² Score                 : {polynomial_r2:.6f}")

# Display sample predictions
results = pd.DataFrame({
    "Actual Price": y_test.values[:10],
    "Linear Prediction": linear_predictions[:10],
    "Polynomial Prediction": polynomial_predictions[:10]
})

print("\n--- SAMPLE PREDICTIONS (First 10) ---")
print(results.to_string(index=False))

# Model comparison
comparison = pd.DataFrame({
    "Model": ["Linear Regression", "Polynomial Regression"],
    "MSE": [linear_mse, polynomial_mse],
    "MAE": [linear_mae, polynomial_mae],
    "R2": [linear_r2, polynomial_r2]
})

print("\n" + "=" * 50)
print("FINAL MODEL COMPARISON")
print("=" * 50)
print(comparison.to_string(index=False))
