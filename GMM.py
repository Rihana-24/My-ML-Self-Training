#Importing libraries
from sklearn.mixture import GaussianMixture #Importing Gaussian Mixture Model
import numpy as np 

# sample data (2D spaces)
X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])

# Initialize model and fit it to the data

gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(X) # estimate model parameters with the EM algorithm mean and covariances of each Gaussian cluster

#Get the cluster labels
labels = gmm.predict(X)
print(labels)# clusters labels

#Get probabilities of each sample belonging to each cluster
probabilities = gmm.predict_proba(X)
print(probabilities)# cluster probabilities

