"""
Quantitative evaluation: Evaluate your system based on one or more evaluation metrics. Choose and motivate which metrics you use.

- Accuracy, Precision, Recall, F1-score, Confusion matrix

Error analysis: Are there specific dialog acts that are more difficult to classify? Are there particular utterances that are hard to classify (for all systems)? And why?

- There are some dialog acts that are more difficult to classify, such as "inform" and "request". For example, the utterance "I want a cheap restaurant" can be classified as "inform" or "request". It is hard to classify because it is a request for a cheap restaurant. The same goes for shared utterances between bye and thankyou.

- The assignment description mentions the following: "In case an utterance was labeled with two different dialog acts, only the first dialog act is used as a label. When performing error analysis (see below) this is a possible aspect to take into account." This could be a reason why some dialog acts are more difficult to classify.

Difficult cases: Come up with two types of ‘difficult instances’, for example utterances that are not fluent (e.g. due to speech recognition issues) or the presence of negation (I don’t want an expensive restaurant). For each case, create test instances and evaluate how your systems perform on these cases.

1. "whats"
2. "tv_noise"

System comparison: How do the systems compare against the baselines, and against each other? What is the influence of deduplication? Which one would you choose for your dialog system?

- 

"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate(model, X, y, label_encoder):
    """
    Evaluate the model based on multiple metrics: accuracy, precision, recall, F1-score, and confusion matrix.
    """
    y_pred = model.predict(X)
    
    # Convert numerical predictions back to original labels
    y_true = label_encoder.inverse_transform(y)
    y_pred = label_encoder.inverse_transform(y_pred)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
    return accuracy, precision, recall, f1, cm

if __name__ == "__main__":
    from ml_1_decision_tree import train_decision_tree_classifier, vectorizer, label_encoder, X_train, X_test, y_train, y_test
    
    # Train the model
    model = train_decision_tree_classifier(vectorizer.fit_transform(X_train), y_train)
    
    # Evaluate the model
    X_test_bow = vectorizer.transform(X_test)
    evaluate(model, X_test_bow, y_test, label_encoder)