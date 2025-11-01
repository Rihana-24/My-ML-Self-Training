#import necessary libraries
from sklearn.svm import SVC #support vector classifier aims to find optimal hyperplane 
from sklearn.model_selection import train_test_split #to split the data into training and testing sets
from sklearn.metrics import accuracy_score, confusion_matrix #to calculate the accuracy of the model
import pandas as pd
import numpy as np

# Load the dataset
#data = pd.read_csv('data.csv') # replace 'data.csv' with the path to your dataset

# sample data (hours study and prior grades vs pass/fail)
X = np.array([[1, 50], [2, 60], [3, 55], [4, 65], [5, 70], [6, 75],[7,80]])# hours and prior grades
# sample labels (pass/fail)
y = np.array([0,0,0,0,1,1,1,1,1,1])

# split data into training and test set 
X_train, X_test, y_train,y_test = train_test_split(X,y, test_size = 0.2 , random_state =42)

# Initialize the SVM model with a linear kernel
model = SVC(kernel='linear')# 'linear' is the default kernel(linear classification) for SVM, RBF,p  poly
model.fit(X_train,y_train)

#Make prediction on the test set 
y_pred = model.predict(X_test)

#Evaluate the model 
accuracy = accuracy_score(y_test,y_pred)
conf_matrix = confusion_matrix(y_test,y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Confusion Matrix:\n{conf_matrix}")
