from sklearn.model_selection import train_test_split
from load_data import load_data, bow_descriptors_labels
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib


# first i create the functions for each classifier that return a trained model

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



#the i create the classifier function that will train the model, test it, and optionally save it

def classfier(choice, X_train, y_train, X_test, y_test, metrics = True, save_model= False): 
    '''
    inputs:
    choice: str ['svm', 'tree', 'logreg']
    X_train, y_train, X_Test, y_test: array
    metrics: bool, True to display classification report, confusion matrix and accuracy score
    save_model: bool: whether the model will be saved in the location spacified in the code
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

        print("\nAccuracy with model " + str(choice) + ": ")
        print(accuracy_score(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # you can comment below out to plot the confusion matrix
        # Plot confusion matrix
        # cm = confusion_matrix(y_test, y_pred)
        # plt.figure(figsize=(10, 8))
        # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
        #             xticklabels=set(y_test), yticklabels=set(y_test))
        # plt.title('Confusion Matrix for ' + str(choice) + ' Classifier')
        # plt.xlabel('Predicted Labels')
        # plt.ylabel('True Labels')
        # plt.show()

    if save_model == True:
        joblib.dump(classifier, 'models/' + choice + '.joblib')

    return classifier


for drop_duplicates in [False, True]:
    #first, loop for duplicated or not and get the dataframe of the sentence/label and the descriptors 
        sentence_label_df = load_data('../data/dialog_acts.dat', drop_duplicates=drop_duplicates)
        X, y = bow_descriptors_labels(sentence_label_df, deduplicated=drop_duplicates, save = True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1, stratify=y)

    #then run every model. we only save the model for the dataframe without the deduplication, because
    #that is when we saw that it performs best
        for classifier_type in ['svm', 'tree', 'logreg']:
            classfier_1 = classfier(classifier_type, X_train, y_train, X_test, y_test, metrics = True, save_model= not drop_duplicates)


'''
the grid search I have commented out tests a lot of different 
parameters for the SVM (SVMs are very sensitive to parameter change)
You can use it on the dataset we have to check if sth works better, 
but I have in the later part chosen the one that worked the best for my
descriptors
'''
'''
from sklearn.model_selection import GridSearchCV

param_grid = [
    {
        'kernel': ['linear'],
        'C': [0.1, 1, 10, 100]
    },
    {
        'kernel': ['poly'],
        'C': [1, 10],
        'degree': [2, 3]
    },
    {
        'kernel': ['rbf'],
        'C': [0.1, 1, 10],
        'gamma': [0.01, 0.1, 1]
    },
    {
        'kernel': ['sigmoid'],
        'C': [1],
        'gamma': [0.01],
    }
]

svm = SVC(decision_function_shape='ovr', random_state=1)
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', verbose=3)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

with open("svm_grid_search_results.txt", "w") as file:
    file.write(f"Best parameters: {grid_search.best_params_}\n")
    file.write(f"Best cross-validation accuracy: {grid_search.best_score_}\n")
    file.write(f"Test set accuracy: {accuracy}\n") 
    
    # Write all cross-validation results (accuracy for each parameter combination)
    file.write("All cross-validation results:\n")
    means = grid_search.cv_results_['mean_test_score']
    stds = grid_search.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
        file.write(f"Accuracy: {mean:.3f} (+/- {std:.3f}) for {params}\n")
'''
'''
I hereby suggest SVM with the best conditions I met in the grid search
You can comment out and use whomever you wish or try which is the best
Or even write in the report that we did a grid search for different parameters
but we can do that together later
'''