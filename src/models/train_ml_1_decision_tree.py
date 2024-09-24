"""
Train the Decision Tree Classifier
"""

from sklearn.tree import DecisionTreeClassifier
import joblib
from load_data import load_data


# Function to train the Decision Tree Classifier
def train_decision_tree_classifier(data_path, version="original", return_type="dataframe"):
    """
    Decision Tree Classifier
    """
    # Load data
    X_train, y_train = load_data(data_path, version, return_type)

    # Creating the Decision Tree Classifier
    classifier = DecisionTreeClassifier()

    # Training the Decision Tree Classifier
    classifier.fit(X_train, y_train)

    # Returning the Confusion Matrix
    return classifier


# Specify data path
data_path = "data/dialog_acts.dat"

# Train the Decision Tree Classifier
classifier = train_decision_tree_classifier(
    data_path, version="normal", return_type="bow"
)

# Notify that traing is done
print(
    "Training is done, the classifier is saved in the current folder as 'decision_tree_classifier.joblib'"
)

# Save the classifier to a file
joblib.dump(classifier, "models/decision_tree_classifier.joblib")
