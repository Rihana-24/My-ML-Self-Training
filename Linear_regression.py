#import neccessary libraries
from sklearn.linear_model import LinearRegression #importing the linear regression model from sklearn
from sklearn.model_selection import train_test_split #importing the train_test_split function from sklearn
from sklearn.metrics import mean_squared_error #importing the mean_squared_error function from sklearn
import numpy as np #importing the numpy library

#Sample data(house size vs house price)
X = np.array([[1400],[1000],[1700],[1875],[1100],[1550],[2450]])
Y = np.array([[235000],[170000],[340000],[380000],[280000],[320000],[450000]])

#split the data into training and testing sets
X_train, X_test,Y_train,Y_test =train_test_split(X,Y, test_size=0.2,random_state=2)     

# Create and fit the linear regression model
#Initialize and train the model
model = LinearRegression()
model.fit(X_train, Y_train)

# Make predictions on the test set
Y_pred = model.predict(X_test)

# Evaluate the model using metric
# Calculate the mean squared error
print("Predicted Values:",Y_pred)
print("Actual ", Y_test)
mse = mean_squared_error(Y_test, Y_pred)
print("Mean squared error:", mse)



        