# Import libraries
from sklearn.ensemble import IsolationForest
import numpy as np

# Sample data 
# Sample data(normal data points clustered around 0)
X =0.3 * np.random.randn(100, 2)
X_train = np.r_[X + 2, X - 2] # create dataset with  points around 0 and 2 clusters
#create test set data including some outliers
x_test = np.r[X + 2, X - 2, np.random.uniform(low=-6,high =6 ,size = [20,2])]

# Inititalize and train the model
model = IsolationForest(contamination=0.1, random_state=0)
model.fit(X_train)


# Make Predictions on test data -1 anomaly 1 data point that is normal
predictions = model.predict(x_test)
print(predictions)

