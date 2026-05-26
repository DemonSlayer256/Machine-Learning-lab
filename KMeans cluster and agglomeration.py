# Write a program to perform unsupervised K-means clustering techniques
# Write a program to perform agglomerative clustering based on single-linkage, complete-linkage criteria


import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
X = np.vstack([
    np.random.randn(50, 2) * 0.5 + [0, 0],
    np.random.randn(50, 2) * 0.5 + [5, 5],
    np.random.randn(50, 2) * 0.5 + [10, 0]
])


# For messy data or more realistic data, we can use the following
# X = np.vstack([
#     np.random.randn(50, 2) * 2.0 + [0, 0],   # Increased spread
#     np.random.randn(50, 2) * 2.0 + [5, 5],
#     np.random.randn(50, 2) * 2.0 + [10, 0]
# ])
# K-Means Clustering
k = int(input("Enter the value of K for clustering:"))
# Number of clusters
kmeans = KMeans(n_clusters=k, random_state=42)
labels_kmeans = kmeans.fit_predict(X)

# Agglomerative Clustering - Single Linkage
agg = int(input("Enter the agglomerative cluster value: "))
agg_single = AgglomerativeClustering(n_clusters=agg, linkage='single')
labels_single = agg_single.fit_predict(X)

# Agglomerative Clustering - Complete Linkage
agg_complete = AgglomerativeClustering(n_clusters=agg, linkage='complete')
labels_complete = agg_complete.fit_predict(X)

# Plotting all results
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Plot K-means
axs[0, 0].scatter(X[:, 0], X[:, 1], c=labels_kmeans, cmap='viridis')
axs[0, 0].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                  s=300, c='red', marker='X')
axs[0, 0].set_title('K-Means Clustering')

# Plot Agglomerative Single-Linkage
axs[0, 1].scatter(X[:, 0], X[:, 1], c=labels_single, cmap='viridis')
axs[0, 1].set_title('Agglomerative Clustering\nSingle Linkage')

# Plot Agglomerative Complete-Linkage
axs[1, 0].scatter(X[:, 0], X[:, 1], c=labels_complete, cmap='viridis')
axs[1, 0].set_title('Agglomerative Clustering\nComplete Linkage')

# Optional: Empty plot for layout
axs[1, 1].axis('off')

# Common labels
for ax in axs.flat:
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

plt.tight_layout()
plt.show()
