import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from joblib import dump, load
import os

# Get the absolute path of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Go up one level to the project root
project_root = os.path.dirname(script_dir)

# Load and split data
def load_and_split_data(
    data_filename,
    version="original",
    return_type="dataframe",
    test_size=0.15,
    random_state=42,
    save_dir=None,
):
    """
    Load data, preprocess it, and split into train and test sets.

    Args:
    data_filename (str): Filename of the data file (should be in the 'data' directory)
    version (str): 'original' or 'deduplicated'
    return_type (str): 'dataframe' or 'bow'
    test_size (float): Proportion of the dataset to include in the test split
    random_state (int): Random state for reproducibility
    save_dir (str): Directory to save the split data (default is 'data' in project root)

    Returns:
    Depending on return_type:
    - 'dataframe': (df_train, df_test)
    - 'bow': (X_train_bow, y_train, X_test, y_test)
    """
    # Set default save_dir if not provided
    if save_dir is None:
        save_dir = os.path.join(project_root, "data")

    # Create the save directory if it does not exist
    os.makedirs(save_dir, exist_ok=True)

    # Construct the full path to the data file
    data_path = os.path.join(project_root, "data", data_filename)

    # Check if split data already exists
    train_path = os.path.join(save_dir, f"train_{version}.joblib")
    test_path = os.path.join(save_dir, f"test_{version}.joblib")

    # Check if split data already exists
    if os.path.exists(train_path) and os.path.exists(test_path):
        print("Loading existing split data...")
        df_train = load(train_path)
        df_test = load(test_path)
    
    # If split data does not exist, create it
    else:
        print("Creating new split data...")
        # Load data
        try:
            with open(data_path, "r", encoding="utf-8") as file:
                data = file.read().split("\n")
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {data_path}")

        # Create Pandas DataFrame
        label, sentence = [], []
        for line in data:
            if line:
                parts = line.split()
                label.append(parts[0])
                sentence.append(" ".join(parts[1:]))
        df = pd.DataFrame({"sentence": sentence, "label": label})

        # Deduplicate if required
        if version == "deduplicated":
            df = df.drop_duplicates(subset=["sentence"])
            
        # Print class distribution
        print("Class distribution:")
        class_dist = df['label'].value_counts()
        print(class_dist)
        print("\nClasses with only one sample:")
        print(class_dist[class_dist == 1])

        # Initialize LabelEncoder
        label_encoder = LabelEncoder()
        df["numerical_label"] = label_encoder.fit_transform(df["label"])
        
        # Identify classes with only one sample
        problematic_classes = class_dist[class_dist == 1].index.tolist()
        if problematic_classes:
            print(f"\nWarning: The following classes have only one sample: {problematic_classes}")
            print("These will be removed to allow stratification.")
            df = df[~df['label'].isin(problematic_classes)]
            print(f"Rows remaining after removal: {len(df)}")
            
        # Split the data
        try:
            df_train, df_test = train_test_split(
                df, test_size=test_size, random_state=random_state, stratify=df["label"]
            )
        except ValueError as e:
            print(f"Error during split: {str(e)}")
            print("Attempting split without stratification...")
            df_train, df_test = train_test_split(
                df, test_size=test_size, random_state=random_state
            )

        # Save the split using joblib
        dump(df_train, train_path)
        dump(df_test, test_path)

    # Return either DataFrame or Bag-of-Words representation
    if return_type == "dataframe":
        return df_train, df_test
    
    # Return Bag-of-Words representation
    elif return_type == "bow":
        vectorizer = CountVectorizer()
        X_train_bow = vectorizer.fit_transform(df_train["sentence"])
        X_test = vectorizer.transform(df_test["sentence"])
        y_train = df_train["numerical_label"]
        y_test = df_test["numerical_label"]

        # Save vectorizer
        dump(vectorizer, os.path.join(save_dir, f"vectorizer_{version}.joblib"))

        return X_train_bow, y_train, X_test, y_test
    
    # Invalid user input
    else:
        raise ValueError("Invalid 'return_type'. Choose 'dataframe' or 'bow'.")


# Load data, split into train and test sets, and save the split data
if __name__ == "__main__":
    df_train, df_test = load_and_split_data(
        "dialog_acts.dat", version="original", return_type="dataframe"
    )
    X_train_bow, y_train, X_test, y_test = load_and_split_data(
        "dialog_acts.dat", version="deduplicated", return_type="bow"
    )
    print("Data loaded successfully!")
    print(f"Train set shape: {df_train.shape}")
    print(f"Test set shape: {df_test.shape}")
    print(f"BOW train set shape: {X_train_bow.shape}")
    print(f"BOW test set shape: {X_test.shape}")
