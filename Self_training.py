# Import necesseray libraries

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score  
import numpy as np

# Generate a synthetic dataset
X,y = make_classification(n_samples=200, n_features=5,random_state=42)

# Split the dataset into training and testing sets
X_labeled, X_test, y_labeled,y_test = train_test_split(X, y, test_size=0.7, random_state=42)

# Initialize and train the model with labeled data

model = RandomForestClassifier(random_state=42)
model.fit(X_labeled,y_labeled)

# Perform selef-training on unlabeled data 
for _ in range(5): # repeat the process 5 times for iterative training
    # Predict probabilities on the unlabeled
    probs = model.predict_proba(X_test)
    high_confidence_idx =np.where(np.max(probs, axis=1)> 0.9)[0] # Select samples with high confidence
    
    # Add high-confidence predictions to labeled data
    X_labeled = np.vstack([X_labeled,X_test[high_confidence_idx]])
    y_labeled = np.hstack([y_labeled,y_test[high_confidence_idx]])
    
    #Remove confident samples from the unlabeled dataset
    X_test = np.delete(X_test,high_confidence_idx,axis=0)
    #Re-train the model on the expanded labeled dataset 
    model.fit(X_labeled,y_labeled)
    
#Final evaluation on test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)