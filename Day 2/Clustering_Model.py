import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Generate random 2D points
np.random.seed(0)
X = np.random.rand(300, 2) * 10

kmeans = KMeans(n_clusters=3, random_state=0, n_init='auto')
kmeans.fit(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

df = pd.DataFrame(X, columns=['x', 'y'])
df['cluster'] = labels

plt.figure(figsize=(8, 6))
scatter = plt.scatter(df['x'], df['y'], c=df['cluster'], cmap='viridis', s=50, alpha=0.8)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.9, marker='X', label='Centroids')
plt.title('K-Means Clustering of Random 2D Points')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.colorbar(scatter, label='Cluster')
plt.legend()
plt.grid(True)
plt.show()