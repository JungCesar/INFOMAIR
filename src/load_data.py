"""
Load data and give option to how to return it: as a pandas DataFrame or Bag of Words (BOW) representation
"""

def load_data(data_path, return_type='dataframe'):
    """
    Load data and give option to how to return it: as a pandas DataFrame or Bag of Words (BOW) representation
    """
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    import pandas as pd
    
    # Open data file in read mode
    with open(data_path, 'r') as file:
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
    
    if return_type == 'dataframe':
        return df
    elif return_type == 'bow':
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
        
        return X_train_bow.toarray()
    else:
        raise ValueError("Invalid return_type. Choose 'dataframe' or 'bow'")