# Practical 8: Dimensionality Reduction using PCA with SVM
import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 1. Load Dataset
path = "/home/avcoe/wdbc.csv" if os.path.exists("/home/avcoe/wdbc.csv") else "datasets/wdbc.csv"
df = pd.read_csv(path)
if 'id' in df.columns:
    df = df.drop('id', axis=1)

X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# 2. Standardization & Split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 3. SVM Classifier BEFORE PCA
svm_before = SVC(kernel="rbf")
svm_before.fit(X_train, y_train)
acc_before = accuracy_score(y_test, svm_before.predict(X_test))

# 4. Apply PCA (Retaining 95% Variance)
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# 5. SVM Classifier AFTER PCA
svm_after = SVC(kernel="rbf")
svm_after.fit(X_train_pca, y_train)
acc_after = accuracy_score(y_test, svm_after.predict(X_test_pca))

# 6. Summary Comparison
print("--- PCA DIMENSIONALITY REDUCTION SUMMARY ---")
print(f"Features Before PCA : {X.shape[1]}  --> Accuracy: {acc_before * 100:.2f}%")
print(f"Features After PCA  : {X_train_pca.shape[1]}  --> Accuracy: {acc_after * 100:.2f}%")
print(f"Total Variance Kept : {pca.explained_variance_ratio_.sum() * 100:.2f}%")
