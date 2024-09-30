"""
Load data and give option to how to return it: as a pandas DataFrame or Bag of Words (BOW) representation
"""

def load_data(data_path, drop_duplicates=False):
    """
    Load data and give option to how to return it: as a pandas DataFrame or Bag of Words (BOW) representation
    """
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
    if drop_duplicates == True:
        df = df.drop_duplicates()
    
    return df
    