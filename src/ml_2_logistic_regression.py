"""
Logistic Regression Classification Model
"""
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

def logistic_regression(X_train, X_test, y_train, y_test):
    """
    Logistic Regression Classification Model
    """
    # Train the model
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)

    # Make predictions
    y_pred = log_reg.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    return log_reg

data_path = "../data/dialog_acts.dat"
df = pd.read_csv(data_path)
#print(df)
X = []
y = []
for index, row in df.iterrows():
    text = row
    label, content = text.split(' ', 1)  # Split by first space
    print(label)
    y.append(label)
    
    #X.append(sentence[1:])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
classifier = logistic_regression(X_train, X_test, y_train, y_test)
# Train the Decision Tree Classifier


# Notify that traing is done
print(
    "Training is done, the classifier is saved in the current folder as 'logistic_regression_classifier.joblib'"
)

# Save the classifier to a file
joblib.dump(classifier, "models/logistic_regression_classifier.joblib")
# # Example usage
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# logistic_regression(X_train, X_test, y_train, y_test)
