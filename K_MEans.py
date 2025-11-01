#Importing Libraries
from sklearn.cluster import KMeans
import numpy as np

# Sample Data
X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0]])
y = np.array([0, 0, 0, 1, 1, 1])

# Initialize and fit the model

Kmeans =KMeans(n_clusters=2, random_state=42)# data should be partionned to 2 clusters(groups)
Kmeans.fit(X)# fit the kmeans model on the data


# Get the cluster centers and labels
centroids = Kmeans.cluster_centers_ # centers after training, each centroid represents the mean of the data points in the cluster
labels = Kmeans.labels_ # labels assigned to each data point based on their distance from the centroids 

print("Cluster Centers:", centroids)
print("Cluster Labels:", labels)
print("Number of Clusters:", Kmeans.n_clusters)
print("Number of Iterations:", Kmeans.n_iter_)
print("Inertia:", Kmeans.inertia_)