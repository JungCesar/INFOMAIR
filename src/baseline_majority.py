from lookup_table import lookup_table

def baseline_model_majority(_):
    """Baseline model always returns the majority class"""
    # return data.value_counts().idxmax()
    # return "inform"
    return lookup_table(6)

# Example usage with user input
while True:
    user_input = input(">")
    result = baseline_model_majority(user_input)
    print(result)