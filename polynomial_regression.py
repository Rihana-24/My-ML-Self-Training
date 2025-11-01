#import necesseray libraries
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

#Sample data( experience vs salary)
X = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])
Y = np.array([[1000],[1500],[2000],[2500],[3000],[3500],[4000],[4500],[5000],[5500]])

#Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#Create a polynomial regression model with degree 2 , transform features into polynomial features
poly = PolynomialFeatures(degree=2)# degree determines the polynomial components
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train_poly, Y_train)# captures non linear relationship between features and target variable

#Make predictions on the test set
Y_pred = model.predict(X_test_poly)
# Evaluate the model
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)
print("Predicted Values:", Y_pred)

