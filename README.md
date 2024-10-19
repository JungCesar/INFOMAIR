# INFOMAIR
GitHub Repository for INFOMAIR: Methods in Artificial Intelligence Research Course at Utrecht University (UU) Group 17

By Christos Loannidis, Jort Koedijk, Luc Minnee and Julius Bijkerk


1. In the data folder all the datasets exist, included the restaurant dataframe after the additional columns.
2. In the src folder:
    2a. In models we have saved the trained models, as well as the vectorizer that we have trained, so that it 
    can work in real time.
    2b. In baselines.py there exist the functions to create the two baseline models
    2c. load_data.py has some utility functions that will be used in classifier.py:
        - preprocess_text(): that applies the lowercasing, lemmatization and removal of special characters both for train/test and for input text.
        - load_data(): returns the data in a pandas dataframe of sentence - act label form, with the argument drop_duplicates identifying if the duplicates will be dropped.
        - bow_descriptors_labels(): creates a saves a vectorizer based on the dataset created in load_data to create a descriptor for an utterance.
    2d. classifier.py is the file where the experimentation with the models (svm, logistic regression, decision tree) happens. 
        - the functions svm_classifier(), decision_tree_classifier(), logistic_regression() input training data that are created by bow_descriptors_labels(),  
        - the classifier() function trains the specified in an argument model, and optionally prints out the metrics (accuracy score, precision etc). It also optionally saves the model itself for the real time communiaction. It also runs for drop_duplicates=True and False as an argument
        - In the end there is a loop that iterates through every classifier and either with dropped duplicates or not and outputs results.
        - Commented out is the code for the grid search for the SVM.
    2e. keyword_mapping.py has the helper functions for finding the keywords both for the basic as well as the additional requirements, and querying the database:
        - initiate_category_dict(): this function extracts unique categories from the restaurant database for food, pricerange, and area. It returns these categories in a dictionary format, which will later be used to match user preferences with available options in the dataset.
        - match_edit_dist(): this function calculates the similarity between an input_token (a user’s preference) and a list of preference_keywords (the possible categories in the database) using the Levenshtein distance algorithm. If use_lev is set to True, the edit distance is computed and compared against a threshold (edit_dist_threshold). The function returns the closest match based on the minimum distance found.
        - extract_preferences(): This function extracts user preferences from inform_text (the user’s input) and compares them with categories available in the restaurant database (preference_categories_dict). It splits the input text into individual words and attempts to match them with the database categories using the match_edit_dist function. The result is a dictionary of user preferences with categories like food, pricerange, and area.
        - query_restaurant(): this function queries the restaurant database using the extracted user preferences (preferences). Based on the version argument, it can filter the database using either equality (eq) or inequality (ineq). The function returns either a list of matching restaurant names or a filtered DataFrame of restaurant records, depending on the output parameter.
    2f. reasoning.py it includes the helper function inference_rules() that applies the reasoning system for 1c.
    2g. Has the main funciton as well as the configurability and the system transition function. more are commented in the code.
