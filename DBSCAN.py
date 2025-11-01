# import libraries
from sklearn.cluster import DBSCAN
import numpy as np


# sample data 
X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])


# Initialize the DBSCAN model
model= DBSCAN(eps=3, min_samples=2)
model.fit(X)

# Get the cluster labels(-1 indicates noise)
labels = model.labels_
print('Labels:', labels)
