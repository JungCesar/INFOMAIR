from lookup_table import lookup_table

def baseline_model_majority(_):
    """Baseline model always returns the majority class"""
    # return data.value_counts().idxmax()
    # return "inform"
    return lookup_table(6)