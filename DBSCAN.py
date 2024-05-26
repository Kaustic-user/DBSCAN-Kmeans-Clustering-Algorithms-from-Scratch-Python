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

"""# DBSCAN"""

#DBSCAN Algorithm

def euclidian_distance(pt1,pt2):
  return np.sqrt(np.sum((pt1 - pt2) ** 2))

def get_neighbours(X,pt_idx,eps):
  neighbours = []
  for i in range (0,X.shape[0]):
    if euclidian_distance(X[pt_idx],X[i])<=eps :
      neighbours.append(i)

  return neighbours

def expand_cluster(X,cluster_id,neighbours,pt_idx):
  labels[pt_idx] = cluster_id
  i=0;
  while i<len(neighbours):
    neigh = neighbours[i]

    if labels[neigh]==-1 :
      labels[neigh] = cluster_id  #adding noise to border point

    if labels[neigh]==0 :
      labels[neigh] = cluster_id  #adding point to cluster

      next_neigh = get_neighbours(X,neigh,eps)
      if len(next_neigh) >= min_pts:
        neighbours += next_neigh

    i+=1

labels = np.zeros(df.shape[0]) #-1 will be assigned for noise points,0 for unvisited
eps = 0.2
min_pts = 5
cluster_id=0

for i in range (0,X.shape[0]):
  if labels[i]!=0:
    continue

  #Finding the neighbours
  neighbours = get_neighbours(X, i, eps)

  if len(neighbours)<min_pts:
    labels[i] = -1 #mark as noise point
  else:
    cluster_id += 1
    expand_cluster(X,cluster_id,neighbours,i)

print(np.unique(labels))

unique_labels = np.unique(labels)
plt.figure(figsize=(8, 6))
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    if k == -1:
        col = [0, 0, 0, 1]  # Black for noise

    class_member_mask = (labels == k)

    xy = X[class_member_mask]
    plt.scatter(xy[:, 0], xy[:, 1], marker='o', facecolors=tuple(col), edgecolors='none', s=60, label=f'Cluster {int(k)}' if k != -1 else 'Noise')

plt.title('DBSCAN Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
