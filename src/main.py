
from baseline_majority import baseline_model_majority
from baseline_keywords import baseline_model_keywords


# Main function of the restaurant recommendation system
def main():

    # Welcome message
    print("Welcome to the restaurant recommendation system!")
    

    # Ask for input, until valid input is given
    while True:

        # User input
        user_input = input(
            "Which model would you like to use? Type:\n1. 'bl1' for Baseline 1: Majority Class\n2. 'bl2' for Baseline 2: Keyword Matching\n3. 'ml1' for Machine Learning Model 1: Decision Tree Classifier\n4. 'ml2' for Machine Learning Model 2: Logistic Regression\n5. 'exit' to exit\n>"
        )

        # Baseline 1: Majority Class
        if user_input.lower() == "bl1":
            print("You chose Baseline 1: Majority Class")
            
            while True:
                user_input = input(">")
                result = baseline_model_majority(user_input)
                print(result)

        # Baseline 2: Keyword Matching
        elif user_input.lower() == "bl2":
            print("Baseline 2: Keyword Matching")
            while True:
                user_input = input(">")
                result = baseline_model_keywords(user_input)
                print(result)

        # Machine Learning Model 1: Decision Tree Classifier
        elif user_input.lower() == "ml1":
            print("Machine Learning Model 1: Decision Tree Classifier")
            break

        # Machine Learning Model 2: Support Vector Machine
        elif user_input.lower() == "ml2":
            print("Machine Learning Model 2: Support Vector Machine")
            break

        # Machine Learning Model 3: Logistic Regression
        elif user_input.lower() == "ml3":
            print("Machine Learning Model 3: Logistic Regression")
            break

        # Exit
        elif user_input.lower() == "exit":
            exit()

        # Invalid input
        else:
            print(
                "Invalid input. Please try again, type: 'bl1', 'bl2', 'ml1', 'ml2' or 'exit'"
            )


# Entry point of the program
if __name__ == "__main__":
    main()
