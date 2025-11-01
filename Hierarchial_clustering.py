#import libraries
from scipy.cluster.hierarchy import dendrogram, linkage # linkage perform agglomerative clustering
import matplotlib.pyplot as plt
import numpy as np

# Sample data 

X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0]])

# Perform  hierarchical/agglomerative clustering
Z = linkage(X, method ='ward') # ward = Euclidean distance minimize variance within each cluster

# Plot the dendrogram
plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Data Points')
plt.ylabel('Distance')
plt.show()