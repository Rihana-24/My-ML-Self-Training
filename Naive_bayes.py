# import libraries
from sklearn.naive_bayes import GaussianNB #  is a Naive Bayes classifier use for continuous data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

 # sample data (hours study and prior grades vs pass/fail)
X = np.array([[1, 50], [2, 60], [3, 55], [4, 65], [5, 70], [6, 75],[7,80]])# hours and prior grades
# sample labels (pass/fail)
y = np.array([0,0,0,0,1,1,1,1,1,1])

# split data into training and testing set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Initialize the Gaussian Naive Bayes classifier
model = GaussianNB()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")




