"""
Quantitative evaluation: Evaluate your system based on one or more evaluation metrics. Choose and motivate which metrics you use.

- Accuracy, Precision, Recall, F1-score, Confusion matrix

Error analysis: Are there specific dialog acts that are more difficult to classify? Are there particular utterances that are hard to classify (for all systems)? And why?

- There are some dialog acts that are more difficult to classify, such as "inform" and "request". For example, the utterance "I want a cheap restaurant" can be classified as "inform" or "request". It is hard to classify because it is a request for a cheap restaurant. The same goes for shared utterances between bye and thankyou.


Difficult cases: Come up with two types of ‘difficult instances’, for example utterances that are not fluent (e.g. due to speech recognition issues) or the presence of negation (I don’t want an expensive restaurant). For each case, create test instances and evaluate how your systems perform on these cases.

1. "whats"
2. "tv_noise"

System comparison: How do the systems compare against the baselines, and against each other? What is the influence of deduplication? Which one would you choose for your dialog system?

- 

"""

def evaluate(model, X, y):
    """
    Evaluate the model based on accuracy (and other metrics: to be implemented!)
    """
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    return accuracy