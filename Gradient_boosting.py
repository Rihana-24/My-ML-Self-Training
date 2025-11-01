#import necessary libraries
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import numpy as np

# sample data (hours study and prior grades vs pass/fail)
X = np.array([[1, 50], [2, 60], [3, 55], [4, 65], [5, 70], [6, 75],[7,80]])# hours and prior grades
# sample labels (pass/fail)
y = np.array([0,0,0,0,1,1,1,1,1,1])

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model and fit to the training data
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,random_state=42)
model.fit(X_train, y_train)



# Make Predictions
y_pred = model.predict(X_test)
print("Predicted labels: ", y_pred)
print("Actual labels: ", y_test)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)
# Calculate confusion matrix
confusion = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: \n", confusion)
