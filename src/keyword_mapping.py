import Levenshtein as lev
import pandas as pd
import random



#database's restaurant categories
preference_categories = { 
    "food"  : ["british", "modern european", "italian", "romanian", "seafood", "chinese",
    "steakhouse", "asian oriental", "french", "portuguese", "indian", "moroccan",  
    "vietnamese", "korean", "swiss", "thai", "spanish", "european", "turkish",  
    "fusion", "tuscan", "polynesian", "african", "mediterranean", "jamaican", 
    "north american", "gastropub", "australasian", "lebanese", "cuban", 
    "persian", "japanese", "catalan"], 

    "pricerange" : ["cheap", "moderate", "expensive"],

    "area" : ["north", "south", "east", "west", "centre"]
}


#edit distance matching
def match_edit_dist(input_token, preference_keywords, edit_dist_threshold):
    #input token is the candidate keyword/preference 
    #db_category is the possible preference category describing food/pricerage/area

    token_match = None
    min_dist = 1000.00

    for category_token in preference_keywords:
        edit_dist = lev.distance(input_token, category_token)
        # print('edit dist ' + str(edit_dist) + ' between ' + category_token + ' and ' + input_token)
        if edit_dist < min_dist and edit_dist <= edit_dist_threshold:
            token_match = category_token
            min_dist = edit_dist        
        # print('match : ' + str(token_match) + 'with distance ' + '0')
    return token_match


def extract_preferences(inform_text):
    #text that is classified as inform will be the input
    #the output is a dictionary in the format of 'preferences' below, with the found preferences per category 
    preferences = {"food": None,
                   "pricerange": None, 
                   "location": None}
    inform_text = inform_text.split()

    for category, keywords in preference_categories.items(): 
        for word in inform_text: #iterates every category and sees if a match is found on the keywords 
            # (ask if we can vectorize the words to find synonyms or just settle with edit distance)
            closest_match = match_edit_dist(word, keywords, 1)
            if closest_match:
                preferences[category] = closest_match
                # print('closest match '+closest_match+ ' in category ' + category)
                break  
    return preferences


def get_restaurant(preferences, resaurant_info_df):

    query_string = ' & '.join([f"{key} == '{value}'" for key, value in preferences.items() if value])
    if query_string:
        # print(query_string)
        return (resaurant_info_df.query(query_string))['restaurantname'].tolist()
    else:
        # print('no query')
        return resaurant_info_df['restaurantname'].tolist()

# csv_file = pd.read_csv("data/restaurant_info.csv")
# preferences_list = extract_preferences("want italian food but also cheap")
# print(get_restaurant(preferences_list, csv_file))
