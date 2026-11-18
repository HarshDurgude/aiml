# Practical 8: Dimensionality Reduction using PCA (Principal Component Analysis) with SVM
# Dataset: wdbc.csv (Wisconsin Diagnostic Breast Cancer)
# AIML Laboratory

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
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
    return f"/home/avcoe/{filename}"


# 1. Load Dataset
dataset_path = get_dataset_path("wdbc.csv")
print(f"Loading dataset from: {dataset_path}")
df = pd.read_csv(dataset_path)

print("First 5 rows:")
print(df.head())

print("\nDataset Shape:", df.shape)

# Drop id if present
if "id" in df.columns:
    df = df.drop("id", axis=1)

# 2. Separate Features and Target
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

print("\nNumber of features before PCA:", X.shape[1])
print("\nClass Distribution:")
print(y.value_counts())
print("\nM = Malignant, B = Benign")

# 3. Split Dataset into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"\nTraining Samples: {X_train.shape[0]}")
print(f"Testing Samples : {X_test.shape[0]}")

# 4. Standardize Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Classification BEFORE PCA using SVM
model_before = SVC(kernel="rbf", C=1.0, gamma="scale")
model_before.fit(X_train_scaled, y_train)
y_pred_before = model_before.predict(X_test_scaled)
accuracy_before = accuracy_score(y_test, y_pred_before)

print("\n" + "=" * 50)
print("CLASSIFICATION BEFORE PCA (SVM)")
print("=" * 50)
print(f"Number of Features : {X_train_scaled.shape[1]}")
print(f"Accuracy           : {accuracy_before:.4f} ({accuracy_before * 100:.2f}%)")
print("\nClassification Report:\n", classification_report(y_test, y_pred_before))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_before))

# 6. Apply PCA with 2 Components for Visualization
scaler_vis = StandardScaler()
X_scaled_vis = scaler_vis.fit_transform(X)
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled_vis)

print("\n" + "=" * 50)
print("PCA 2D VISUALIZATION METRICS")
print("=" * 50)
print(f"PC1 Variance Ratio: {pca_2d.explained_variance_ratio_[0]:.4f}")
print(f"PC2 Variance Ratio: {pca_2d.explained_variance_ratio_[1]:.4f}")
print(f"Total Explained by 2 PCs: {np.sum(pca_2d.explained_variance_ratio_):.4f}")

# 7. PCA 2D Scatter Plot
plt.figure(figsize=(8, 6))
label_names = {"M": "Malignant", "B": "Benign"}
for label in y.unique():
    plt.scatter(
        X_pca_2d[y == label, 0],
        X_pca_2d[y == label, 1],
        label=label_names.get(label, str(label)),
        alpha=0.7
    )
plt.xlabel("Principal Component 1 (PC 1)")
plt.ylabel("Principal Component 2 (PC 2)")
plt.title("PCA 2D Visualization of WDBC Dataset")
plt.legend(title="Diagnosis")
plt.grid(True)
plt.tight_layout()
plt.show()

# 8. PCA for Classification (Retain 95% Variance)
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

number_of_pca_features = X_train_pca.shape[1]

print("\n" + "=" * 50)
print("PCA FOR CLASSIFICATION (95% Variance Retained)")
print("=" * 50)
print(f"Features Before PCA : {X_train_scaled.shape[1]}")
print(f"Features After PCA  : {number_of_pca_features}")
print(f"Total Variance Kept : {np.sum(pca.explained_variance_ratio_):.4f}")

# 9. Classification AFTER PCA using SVM
model_after = SVC(kernel="rbf", C=1.0, gamma="scale")
model_after.fit(X_train_pca, y_train)
y_pred_after = model_after.predict(X_test_pca)
accuracy_after = accuracy_score(y_test, y_pred_after)

print("\n" + "=" * 50)
print("CLASSIFICATION AFTER PCA (SVM)")
print("=" * 50)
print(f"Number of Features : {number_of_pca_features}")
print(f"Accuracy           : {accuracy_after:.4f} ({accuracy_after * 100:.2f}%)")
print("\nClassification Report:\n", classification_report(y_test, y_pred_after))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_after))

# 10. Final Comparison
difference = accuracy_after - accuracy_before
print("\n" + "=" * 50)
print("FINAL SUMMARY COMPARISON")
print("=" * 50)
print(f"Original Features   : {X.shape[1]}")
print(f"Features After PCA  : {number_of_pca_features}")
print(f"Accuracy Before PCA : {accuracy_before:.4f}")
print(f"Accuracy After PCA  : {accuracy_after:.4f}")
print(f"Accuracy Difference : {difference:+.4f}")

# 11. Bar Chart Comparison
plt.figure(figsize=(6, 5))
methods = ["Before PCA", "After PCA"]
accuracies = [accuracy_before, accuracy_after]
bars = plt.bar(methods, accuracies, color=['skyblue', 'salmon'], edgecolor='black')
plt.ylabel("Accuracy")
plt.title("SVM Accuracy Before vs After PCA")
plt.ylim(min(accuracies) - 0.05, 1.0)

for bar, acc in zip(bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        acc + 0.005,
        f"{acc:.4f}",
        ha="center",
        fontweight='bold'
    )
plt.tight_layout()
plt.show()
