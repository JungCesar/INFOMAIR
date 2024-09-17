"""
Logistic Regression Classification Model
"""

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

# # Example usage
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# logistic_regression(X_train, X_test, y_train, y_test)
