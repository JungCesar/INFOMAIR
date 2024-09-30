from sklearn.model_selection import train_test_split
from load_data import load_data
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import GridSearchCV


sentence_label_df = load_data('./data/dialog_acts.dat', drop_duplicates=True)

def bow_descriptors_labels(sentence_label_df):


    label_encoder = LabelEncoder()
    
    #transform the labels to numerical values
    sentence_label_df['numerical_label'] = label_encoder.fit_transform(sentence_label_df['label'])

    #create bow descriptors in sparse form
    vectorizer = CountVectorizer()
    X_bow = vectorizer.fit_transform(sentence_label_df["sentence"])

    return X_bow, sentence_label_df['numerical_label'] 

def svm_classifier(X_train, y_train):
    svm = SVC(kernel= 'linear', C=1 , decision_function_shape='ovr', random_state=1)
    # svm = SVC(kernel= 'linear', C=10, decision_function_shape='ovr', random_state=1)
    # svm = SVC(kernel= 'poly', C=10, degree=3, decision_function_shape='ovr', random_state=1)
    # svm = SVC(kernel= 'rbf', C=10, gamma=0.1, decision_function_shape='ovr', random_state=1)
    svm.fit(X_train, y_train)
    return svm


def decision_tree_classifier(X_train, y_train):
    decisiontree = DecisionTreeClassifier()
    decisiontree.fit(X_train, y_train)
    return decisiontree


def logistic_regression(X_train, y_train):
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    return log_reg

# def feedforward_nn(X, y, hidden_layers, num_iterations, learning_rate):

#     weights, biases = initialize_weights(X, hidden_layers)
#     for _ in range(num_iterations):
#         activations = forward_pass(X, weights, biases)
#         loss = compute_loss(y, activations)
#         gradients = backward_pass(X, y, activations, weights, biases)
#         weights, biases = update_weights(weights, biases, gradients, learning_rate)
#     return weights, biases

# y_pred = svm.predict(X_test)

def classfier(choice, X_train, y_train, X_test, y_test, metrics = True, save_model_loc= None): 
    '''
    inputs:
    choice: str ['svm', 'tree', 'logreg']
    X_train, y_train, X_Test, y_test: array
    metrics: bool, True to display classification report, confusion matrix and accuracy score
    save_model_loc: str: if None, it doesn't save the model. if str, it saves it to the location   
                         models/-chosen name-.joblib
    '''
    if choice == 'svm':
        classifier = svm_classifier(X_train, y_train)
    elif choice == 'tree':
        classifier = decision_tree_classifier(X_train, y_train)
    elif choice == 'logreg':
        classifier = logistic_regression(X_train, y_train)
    else:
        print('Error: Invalid in classifier choice selected. ["svm", "tree", "logreg"]')


    y_pred = classifier.predict(X_test)

    if metrics == True:
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        # print("Confusion Matrix with model " + str(choice) + ": ")
        # print(confusion_matrix(y_test, y_pred))
        print("\nAccuracy with model " + str(choice) + ": ")
        print(accuracy_score(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

    if save_model_loc != None:
        import joblib
        joblib.dump(classifier, 'models/' + save_model_loc + '.joblib')

    return classifier


X, y = bow_descriptors_labels(sentence_label_df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
print(X_train)
classfier_1 = classfier('svm', X_train, y_train, X_test, y_test, metrics = True, save_model_loc= None)




# the grid search I have commented out tests a lot of different 
# parameters for the SVM (SVMs are very sensitive to parameter change)
# You can use it on the dataset we have to check if sth works better, 
# but I have in the later part chosen the one that worked the best for my
# descriptors

# param_grid = [
#     {
#         'kernel': ['linear'],
#         'C': [0.1, 1, 10, 100]
#     },
#     {
#         'kernel': ['poly'],
#         'C': [1, 10],
#         'degree': [2, 3]
#     },
#     {
#         'kernel': ['rbf'],
#         'C': [0.1, 1, 10],
#         'gamma': [0.01, 0.1, 1]
#     },
#     {
#         'kernel': ['sigmoid'],
#         'C': [1],
#         'gamma': [0.01],
#     }
# ]

# svm = SVC(decision_function_shape='ovr', random_state=1)
# grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', verbose=3)
# grid_search.fit(X_train, y_train)
# best_model = grid_search.best_estimator
# y_pred = best_model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)

# with open("svm_grid_search_results.txt", "w") as file:
#     file.write(f"Best parameters: {grid_search.best_params_}\n")
#     file.write(f"Best cross-validation accuracy: {grid_search.best_score_}\n")
#     file.write(f"Test set accuracy: {accuracy}\n") 
    
#     # Write all cross-validation results (accuracy for each parameter combination)
#     file.write("All cross-validation results:\n")
#     means = grid_search.cv_results_['mean_test_score']
#     stds = grid_search.cv_results_['std_test_score']
#     for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
#         file.write(f"Accuracy: {mean:.3f} (+/- {std:.3f}) for {params}\n")


# I hereby suggest SVM with the best conditions I met in the grid search
# You can comment out and use whomever you wish or try which is the best
# Or even write in the report that we did a grid search for different parameters
# but we can do that together later

