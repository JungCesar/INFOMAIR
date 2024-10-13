import joblib as jl
import load_data as ld
import pandas as pd
import keyword_mapping as km
import reasoning as reas
import pyttsx3
import os
import keyboard
import threading
import time

sys_configs = {
    "EARLY_SUGGESTIONS" : False,
    "USE_LEVENSHTEIN" : True,
    "ALLOW_CHANGE_PREF" : True,
    "PRINT_AND_SPEAK" : True
}

print(os.getcwd())
model = jl.load("models/svm.joblib")
vectorizer = jl.load("models/vectorizer.joblib")
restaurant_db = pd.read_csv("../data/updated_restaurant_info.csv")
preference_categories_dict = km.initiate_category_dict(restaurant_db)
user_preferences = {
        "food": None,
        "pricerange": None, 
        "area": None
        }
additional_preferences = {
        'romantic' : ['romantic'],
        'assigned seats' : ['assigned'],
        'children' : ['children'],
        'touristic' : ['touristic']
        }

def classify_input(input_text):
    input_text = ld.preprocess_text(input_text)
    descriptor = vectorizer.transform([input_text])
    label = model.predict(descriptor)[0]
    return label

#tts_engine = pyttsx3.init()
#def print_speak_thread():
threads = []


def stop_speach(engine):
    #print("process has started")
    start_time = time.time()
    while True:
        if keyboard.is_pressed('enter'):
            #print("enter is pressed")
            if engine.isBusy():  
                engine.stop()
                break
        elapsed_time = time.time() - start_time
        if elapsed_time > 5:
            break

def speak_text(text):
    tts_engine = pyttsx3.init()
    stop_thread = threading.Thread(target=stop_speach, args = (tts_engine,))    
    stop_thread.daemon = False  # This ensures the thread will close when the main program exits
    stop_thread.start()
    threads.append(stop_thread)
    if stop_thread.is_alive():
        print("Thread is alive")
    tts_engine.say(text)
    
    
    tts_engine.runAndWait()
    stop_thread.join()
    

   

def print_and_speak(text, speak=True):
    
    print(text)
    

    if speak==True:
        
        speak_text(text)
   
        

def ask_preference(category, addition = "", speak=True): 
    print_and_speak( addition + "What type of " + category + " would you like? Please type 'any' if you have no particular preference.", speak)
    text = input().lower()
    if text == "any":
        return "any"
    label = classify_input(text)
    while label != 6:
        print_and_speak("Please, specify the type of " + category + " that you would prefer.", speak)
        text = input().lower()
        if text == "any":
            return "any"
        label = classify_input(text)
    return text

def update_preferences(text, configurations):
    instance_preferences = km.extract_preferences(text, preference_categories_dict, configurations["USE_LEVENSHTEIN"])
    if all(value is None for value in instance_preferences.values()):
        return ""
    expressed_preference = ""
    for category, value in instance_preferences.items():
        if configurations["ALLOW_CHANGE_PREF"]:  
            if value is not None:
                user_preferences[category] = value
                expressed_preference += f"You specified that you want the restaurant to be {value}. "
        else: 
            if value is not None and user_preferences[category] is None:
                user_preferences[category] = value
                expressed_preference += f"You specified that you want the restaurant to be {value}. "

    return expressed_preference


def is_category_filled(key):
    return bool(user_preferences.get(key))


def give_suggestion(subset, backup_subset, inferred_reason, speak, additional_preferences= {}): 
    additional_preferences_provided = any(additional_preferences.values())
    if subset is None or subset.empty:
        if backup_subset is not None and not backup_subset.empty:
            name = backup_subset["restaurantname"].iloc[0] 
            if additional_preferences_provided:
                print_and_speak( "No restaurants match the given additional preferences. \
                    We will attempt to relax the additional preferences and suggest a restaurant without them.", speak)
            print_and_speak( "I would like to propose " + name +". It has "+ backup_subset["food"].iloc[0]\
                + " food, "+ backup_subset["pricerange"].iloc[0] +" prices and is located in the "+ backup_subset["area"].iloc[0] + ".", speak)
        else:
            print_and_speak( "No restaurants matched with your search, even disregarding the additional restaurants.", speak)
            return False, None
    else:    
        name = subset["restaurantname"].iloc[0] 
        print_and_speak( "I would like to propose " + name +". It has "+ subset["food"].iloc[0]\
            + " food, "+ subset["pricerange"].iloc[0] +" prices and is located in the "+ subset["area"].iloc[0] + ". " + inferred_reason, speak)
        
    print_and_speak( "Are you okay with the aforementioned suggestion?", speak)
    text = input().lower()
    label = classify_input(text)
    if label in [4, 7, 13]:
        agreement = False
    else: 
        agreement = True
    return agreement, name

def confirm_preferences(configurations): 
    recap_string = "You have selected: \n"
    if user_preferences["food"]:
        recap_string+= "-" + user_preferences["food"] + " food. \n"
    if user_preferences["pricerange"]:
        recap_string += "-" + user_preferences["pricerange"]+ " prices.\n"
    if user_preferences["area"]:
        recap_string += "-location in the " + user_preferences["area"]+ "."
    speak=  configurations["PRINT_AND_SPEAK"]
    print_and_speak(recap_string, speak)
    if configurations['ALLOW_CHANGE_PREF'] == True:
        print_and_speak( "Would you like to change any of them? Please state them now, or we will move on.", speak)
        text= input().lower()
        label = classify_input(text)
        if label == 6:
            changed_preference = update_preferences(text, configurations)
            print_and_speak( changed_preference, speak)
        
def offer_early_suggestions(user_preferences, restaurant_db ,configurations):
    if configurations['EARLY_SUGGESTIONS'] == True and \
            any(value not in [None, "any"] for value in user_preferences.values()):
        
        restaurant_subset = km.query_restaurant(user_preferences, restaurant_db, output='df', version='eq')
        agreement, restaurant_name = give_suggestion(restaurant_subset, restaurant_subset, "", configurations["PRINT_AND_SPEAK"])
        if agreement:
            print_and_speak( "I hope you enjoy your time in " + restaurant_name, configurations['PRINT_AND_SPEAK'])
            return True, restaurant_db 
        
        
        restaurant_db = restaurant_db[restaurant_db['restaurantname'] != restaurant_name]
    return False, restaurant_db


def state_transition_function(configurations):
    #reload restaurant db, for restarts and the case that in early_suggestions, restaurants have been poped out
    restaurant_db = pd.read_csv("../data/updated_restaurant_info.csv")
    speak= configurations['PRINT_AND_SPEAK']
    #greet
    #stop_thread_ = stop_thread()
    
    print_and_speak( "Hello, welcome to restaurant recommender!", speak)
    #stop_thread_.join()

    #get response and classify it
    text = input().lower()
    label = classify_input(text)
    if label != 6:
        text = ask_preference("food", "", speak)
    
    if text == "any":
        user_preferences["food"] = "any"
        print_and_speak( "You indicated that you have no particular preference for food type.", speak)
    expressed_preference = update_preferences(text, configurations)

    #we go from category to category until they are filled (any)
    while not is_category_filled("food") :
        text = ask_preference("food", addition = expressed_preference + "I would like to know your food type preference. ", speak=speak)
        if text == "any":
            user_preferences["food"] = "any"
            print_and_speak( "You indicated that you have no particular preference for food type.", speak)
            break
        expressed_preference = update_preferences(text, configurations)

    early_agreement, restaurant_db = offer_early_suggestions(user_preferences, restaurant_db, configurations)
    if early_agreement == True:
        return
        
    while not is_category_filled("pricerange"):
        text = ask_preference("pricerange", addition = expressed_preference + "I would also like to know if you want the restaurant to be cheap, moderate, or expensive. ", speak=speak)
        if text == "any":
            user_preferences["pricerange"] = "any"
            print_and_speak( "You indicated that you have no particular preference for price range.", speak)
            break
        expressed_preference = update_preferences(text, configurations)


    early_agreement, restaurant_db = offer_early_suggestions(user_preferences, restaurant_db, configurations)
    if early_agreement == True:
        return
        
        
    while not is_category_filled("area") :
        text = ask_preference("area", addition = expressed_preference + " Would you like the restaurant to be in the north, south, center, east or west? ", speak=speak)
        if text == "any":
            user_preferences["area"] = "any"
            print_and_speak( "You indicated that you have no particular preference for the location.", speak)
            break
        expressed_preference = update_preferences(text, configurations)
    
    early_agreement, restaurant_db = offer_early_suggestions(user_preferences, restaurant_db, configurations)
    if early_agreement == True:
        return
        
    confirm_preferences(configurations)

    #now we ask for additional requirements
    print_and_speak( "Do you have any additional requirements, such as assigned seating, a romantic atomsphere, a touristic place or a children friendly environment?", speak)
    text = input().lower()
    label= classify_input(text)

    if label in [4, 7, 13] or text in ["none", "no", "any"]:
        print_and_speak( "You indicated that you have no additional preferences.",speak)
        selected_added_pref = {}
        filters_true, filters_false, inferred_reason= {}, {}, "" 
    else:    
        selected_added_pref = km.extract_preferences(text, additional_preferences, configurations["USE_LEVENSHTEIN"])

        if any(selected_added_pref.values()):  
            filters_true, filters_false, inferred_reason = reas.inference_rules(selected_added_pref)
        else:
            while True:
                print_and_speak( "No valid preferences were found. Please specify the additional requirements. Or type 'any' to move on.", speak)
                text = input().lower()
                if text == "any":
                    filters_true, filters_false, inferred_reason = {}, {}, ''
                    break  

                selected_added_pref = km.extract_preferences(text, additional_preferences, configurations["USE_LEVENSHTEIN"])
                if any(selected_added_pref.values()):  
                    filters_true, filters_false, inferred_reason = reas.inference_rules(selected_added_pref)
                    break

    #user preferences query
    restaurant_subset_1 = km.query_restaurant(user_preferences, restaurant_db, output = 'df', version ='eq')
    #additional preferences - equality filter
    restaurant_subset = km.query_restaurant(filters_true, restaurant_subset_1, output = 'df', version ='eq')
    #additional preferences - inequality filter
    restaurant_subset= km.query_restaurant(filters_false, restaurant_subset, output = 'df', version ='ineq')

    agreement, restaurant_name = give_suggestion(restaurant_subset, restaurant_subset_1, inferred_reason, speak, filters_true|filters_false)

    if agreement==True:
        print_and_speak( "I hope you enjoy your time in " + restaurant_name, speak)
    else:
        print_and_speak( "Okay, we will then restart the recommendation procedure. ", speak)    
        user_preferences.update({
        "food": None,
        "pricerange": None, 
        "area": None
        })
        state_transition_function(configurations)



def main(configurations):
    
    state_transition_function(configurations)
    
if __name__ == "__main__":
    main(sys_configs)
