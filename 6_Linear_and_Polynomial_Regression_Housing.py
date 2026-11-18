# Practical 6: Regression Analysis (Linear vs Polynomial Degree 2)
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# 1. Load Dataset
path = "Housing.csv"
df = pd.read_csv(path)

# 2. Features & Target (One-hot encode categorical features)
X = pd.get_dummies(df.drop("price", axis=1), drop_first=True)
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
lin_pred = linear_model.predict(X_test)

# 4. Polynomial Regression (Degree 2 with Scaling)
poly_model = Pipeline([
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("scaler", StandardScaler()),
    ("reg", LinearRegression())
])
poly_model.fit(X_train, y_train)
poly_pred = poly_model.predict(X_test)

# 5. Model Comparison
comparison = pd.DataFrame({
    "Model": ["Linear Regression", "Polynomial Regression"],
    "MSE": [mean_squared_error(y_test, lin_pred), mean_squared_error(y_test, poly_pred)],
    "MAE": [mean_absolute_error(y_test, lin_pred), mean_absolute_error(y_test, poly_pred)],
    "R2 Score": [r2_score(y_test, lin_pred), r2_score(y_test, poly_pred)]
})

print("--- MODEL COMPARISON ---")
print(comparison.to_string(index=False))
