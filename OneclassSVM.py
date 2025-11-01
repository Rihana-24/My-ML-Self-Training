#Import necessary libraries
from sklearn.svm import OneClassSVM
import numpy as np


# Sample data(normal data points clustered around 0)
X =0.3 * np.random.randn(100, 2)
X_train = np.r_[X + 2, X - 2] # create dataset with  points around 0 and 2 clusters
#create test set data including some outliers
x_test = np.r[X + 2, X - 2, np.random.uniform(low=-6,high =6 ,size = [20,2])]

#Initialize and train the OneClassSVM model
model = OneClassSVM(gamma ='auto', nu =0.1)# gamma set the kernel coefficient automatically

model.fit(X_train)

#Predict the labels for the test data(-1 indicates an anomaly , 1 indicates for a normal data point)

y_pred = model.predict(x_test)
# Display the predicted labels
print("Predicted labels:", y_pred)
