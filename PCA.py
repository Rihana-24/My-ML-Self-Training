#import libraries
from sklearn.decomposition import PCA
import numpy as np 


#Sample data (3d space)
X = np.array([[1,2,3],[2,3,4],[3,4,5],5,6,7[5,7,8]])

# Initialize  and fit the model
pca = PCA(n_components=2) # Reduce the dimensionality to 2 components
X_reduced= pca.fit(X)
print("Reduced Data:",X_reduced)# data point transformed to 2d space

print('Explained Variance Ration:',pca.explained_variance_ratio_)# the proportion of variance explained by principal component
