# Practical 7: Customer Segmentation (K-Means & Hierarchical Clustering)
import os
import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans

# 1. Load Dataset & Select Features (Age, Annual Income, Spending Score)
path = "/home/avcoe/Mall_Customers.csv" if os.path.exists("/home/avcoe/Mall_Customers.csv") else "datasets/Mall_Customers.csv"
df = pd.read_csv(path)
X = df.iloc[:, [2, 3, 4]].values

# 2. Elbow Method (WCSS) & Dendrogram
wcss = [KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10).fit(X).inertia_ for k in range(1, 11)]

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method for Optimal K")
plt.xlabel("Clusters (K)")
plt.ylabel("WCSS")

plt.subplot(1, 2, 2)
dendrogram(linkage(X, method='ward'))
plt.title("Hierarchical Clustering Dendrogram")
plt.tight_layout()
plt.show()

# 3. Fit K-Means and Agglomerative Clustering (K=5)
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0, n_init=10)
df['KMeans_Cluster'] = kmeans.fit_predict(X)

hc = AgglomerativeClustering(n_clusters=5, linkage='ward')
df['Hierarchical_Cluster'] = hc.fit_predict(X)

print("--- Clustered Customer Sample (First 5 Rows) ---")
print(df[['CustomerID', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'KMeans_Cluster', 'Hierarchical_Cluster']].head())
