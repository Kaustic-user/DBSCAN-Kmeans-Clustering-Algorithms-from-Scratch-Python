# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

# Generate the noisy circle dataset
X, y = make_circles(n_samples=1000, factor=0.5, noise=0.05, random_state=42)

# Plot the dataset
plt.figure(figsize=(8, 6))
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Class 0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1')
plt.title('Noisy Circle Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()

df = pd.DataFrame(X,columns=['feature1','feature2'])
X=df.values
print(df.shape)


"""# Kmeans"""

#K-MEANS Clustering

def init_centroids(X,k):
  indices = np.random.choice(X.shape[0], k, replace=False)
  return X[indices]

def distance(X,centroids):
  distances = np.zeros((X.shape[0], centroids.shape[0]))
  for i, centroid in enumerate(centroids):
        distances[:, i] = np.linalg.norm(X - centroid, axis=1)
  return distances

def update_centroids(X, labels, k):
    centroids = np.zeros((k, X.shape[1]))
    for i in range(k):
        centroids[i, :] = X[labels == i].mean(axis=0)
    return centroids

def kmeans(X, k, max_iters=100, tol=1e-4):
    centroids = init_centroids(X, k)
    for _ in range(max_iters):
        old_centroids = centroids
        distances = distance(X, centroids)
        labels = np.argmin(distances, axis=1)
        centroids = update_centroids(X, labels, k)
        if np.all(np.abs(centroids - old_centroids) < tol):
            break
    return labels, centroids

# Assume X is your data as a numpy array
k = 3  # Number of clusters
labels, centroids = kmeans(X, k)

# Plotting the results
for i in range(k):
    plt.scatter(X[labels == i, 0], X[labels == i, 1], label=f'Cluster {i+1}')
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', label='Centroids')
plt.legend(loc='upper right')
plt.show()
