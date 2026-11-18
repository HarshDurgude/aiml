# Practical 7: Unsupervised Learning - Clustering (K-Means & Hierarchical Clustering)
# Dataset: Mall_Customers.csv
# AIML Laboratory

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans


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


dataset_path = get_dataset_path("Mall_Customers.csv")
print(f"Loading dataset from: {dataset_path}")
mall_data = pd.read_csv(dataset_path)

print("\n--- First 5 Rows ---")
print(mall_data.head())

print("\n--- Dataset Info ---")
mall_data.info()

# Correlation Heatmap
try:
    corr = mall_data.corr(numeric_only=True)
except TypeError:
    corr = mall_data.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, cbar=True, square=True, fmt='.2f', annot=True, cmap='Reds')
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# Exploratory Visualizations
if "Gender" in mall_data.columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(x="Gender", data=mall_data)
    plt.title("Gender Distribution")
    plt.tight_layout()
    plt.show()

if "Age" in mall_data.columns:
    plt.figure(figsize=(10, 5))
    sns.histplot(mall_data["Age"], bins=20, kde=True)
    plt.title("Age Distribution")
    plt.tight_layout()
    plt.show()

# Extract relevant features: Age, Annual Income, and Spending Score
X = mall_data.iloc[:, [2, 3, 4]].values

# --------------------------------------------------
# 1. K-Means Clustering & Elbow Method
# --------------------------------------------------
print("\nCalculating WCSS for Elbow Method (1 to 10 clusters)...")
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=50, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Elbow Graph
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title("The Elbow Method for Optimal K")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS Value (Inertia)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Building final K-Means Model with 5 clusters
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=0, n_init=10)
y_kmeans = kmeans.fit_predict(X)

# 3D Scatter Plot for K-Means
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
colors = ['red', 'blue', 'green', 'yellow', 'purple']

for i in range(optimal_k):
    ax.scatter(
        X[y_kmeans == i, 0],
        X[y_kmeans == i, 1],
        X[y_kmeans == i, 2],
        s=40,
        color=colors[i],
        label=f"Cluster {i + 1}"
    )

ax.set_xlabel('Age')
ax.set_ylabel('Annual Income (k$)')
ax.set_zlabel('Spending Score (1-100)')
ax.set_title("K-Means 3D Clustering")
ax.legend()
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 2. Hierarchical Clustering (Dendrogram & Agglomerative)
# --------------------------------------------------
print("\nPlotting Dendrogram for Hierarchical Clustering...")
plt.figure(figsize=(12, 7))
linked = linkage(X, method='ward')
dendrogram(linked)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.tight_layout()
plt.show()

# Agglomerative Clustering
hc = AgglomerativeClustering(n_clusters=5, linkage='ward')
y_hc = hc.fit_predict(X)

# 3D Scatter Plot for Hierarchical Clustering
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for i in range(5):
    ax.scatter(
        X[y_hc == i, 0],
        X[y_hc == i, 1],
        X[y_hc == i, 2],
        s=40,
        color=colors[i],
        label=f"Cluster {i + 1}"
    )

ax.set_xlabel('Age')
ax.set_ylabel('Annual Income (k$)')
ax.set_zlabel('Spending Score (1-100)')
ax.set_title("Hierarchical Agglomerative 3D Clustering")
ax.legend()
plt.tight_layout()
plt.show()

print("\nPractical 7 execution completed successfully!")
