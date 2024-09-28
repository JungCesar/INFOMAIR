"""
Logistic Regression Classification Model
"""
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix

import pandas as pd
import joblib

def logistic_regression(X_train, X_test, y_train, y_test):
    """
    Logistic Regression Classification Model
    """
    # Train the model
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    print(X_test[0])
    # Make predictions
    y_pred = log_reg.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    ex = vectorizer.transform(["i want to eat italian food"])
    print(log_reg.predict(ex))
    # Evaluate the model
    

    return log_reg

data_path = "../data/dialog_acts.dat"
with open(data_path) as file:
    df = file.readlines()


X = []
y = []
for row in df:
    #row = str(row)
    row = row.split(" ")
    label = row[0]
    row = " ".join(row[1:])
    X.append(row)
    y.append(label)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)
    #X.append(sentence[1:])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
classifier = logistic_regression(X_train, X_test, y_train, y_test)
# Train the Decision Tree Classifier


# Notify that traing is done
print(
    "Training is done, the classifier is saved in the current folder as 'logistic_regression_classifier.joblib'"
)

# Save the classifier to a file
#joblib.dump(classifier, "models/logistic_regression_classifier.joblib")
#joblib.dump(vectorizer, "models/logistic_regression_classifier_vectorizer.joblib")
# # Example usage
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# logistic_regression(X_train, X_test, y_train, y_test)
