"""
Decision Tree Classifier
"""

# Welcome message
print("The Decision Tree Classifier is training now...")

# Import necessary libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Open data file in read mode
with open('data/dialog_acts.dat', 'r') as file:
    # Step 2: Read the content of the file
    data = file.read().split('\n')

# Create Pandas DataFrame from the data
label = []
sentence = []
for line in data:
    if line > '':
        line = line.split()
        label.append(line[0])
        sentence.append(" ".join(line[1:]))
    
df = pd.DataFrame({'sentence': sentence, 'label': label})
df.head()

# Initialize LabelEncoder to convert labels to numerical values
label_encoder = LabelEncoder()

# Fit and transform the labels to numerical values
df['numerical_label'] = label_encoder.fit_transform(df['label'])

# Split the data into training and testing sets (85% training, 15% testing)
X_train, X_test, y_train, y_test = train_test_split(df["sentence"],df["numerical_label"],test_size=0.15,shuffle=True)

# Initialize CountVectorizer for BOW representation
vectorizer = CountVectorizer()

# Fit and transform X_train to BOW representation
X_train_bow = vectorizer.fit_transform(X_train)

def train_decision_tree_classifier(X_train, y_train):
    """
    Decision Tree Classifier
    """
    # Importing the Decision Tree Classifier
    from sklearn.tree import DecisionTreeClassifier

    # Creating the Decision Tree Classifier
    classifier = DecisionTreeClassifier()

    # Training the Decision Tree Classifier
    classifier.fit(X_train, y_train)
    
    # Returning the Confusion Matrix
    return classifier

# Train the Decision Tree Classifier
classifier = train_decision_tree_classifier(X_train_bow, y_train)

# Notify that traing is done
print("Training is done, you can start using the classifier below.")

# Joblib is a set of tools to provide lightweight pipelining in Python
import joblib

# Save the classifier to a file
joblib.dump(classifier, 'models/decision_tree_classifier.joblib')