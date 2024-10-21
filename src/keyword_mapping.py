import Levenshtein as lev
import pandas as pd
import random



#get every category in the database for food, area and pricerange to try to find matches
def initiate_category_dict(restaurant_info_df):

    preference_categories = { 
    "food"  : restaurant_info_df['food'].unique().tolist(),
    "pricerange" : restaurant_info_df['pricerange'].unique().tolist(),
    "area" : restaurant_info_df['area'].dropna().unique().tolist()
}
    return preference_categories


#edit distance matching for a single category
def match_edit_dist(input_token, preference_keywords, edit_dist_threshold, use_lev= True):
    #input token is the candidate keyword/preference 
    #db_category is the possible preference category describing food/pricerage/area

    if not use_lev:
        return input_token if input_token in preference_keywords else None
    
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


def extract_preferences(inform_text, preference_categories_dict, use_lev =True):
    #text that is classified as inform will be the input
    #preference_categories_dict: key: (str)feature, value: list of possible values
    #the output is a dictionary in the format of 'preferences' below, with the found preferences per category 
    preferences = {category: None for category in preference_categories_dict.keys()}
    
    if "modern european" in inform_text and "food" in preferences.keys():
        preferences["food"] = "modern european"
        preference_categories_dict["food"].remove("european") #to avoid overlap

    inform_text = inform_text.split()

    for category, keywords in preference_categories_dict.items(): 
        for word in inform_text: #iterates every category and sees if a match is found on the keywords 
            # (ask if we can vectorize the words to find synonyms or just settle with edit distance)
            # print(word, keywords)
            closest_match = match_edit_dist(word, keywords, 1, use_lev)
            if closest_match:
                preferences[category] = closest_match
                # print('closest match '+closest_match+ ' in category ' + category)
                break  
    return preferences



def query_restaurant(preferences, restaurant_info_df, output = 'list', version ='eq'):

    '''
    lookup to the database for the criteria extracted
    inputs:
    preferenfes: dict, key: column name, value: the value for that feature in the df
    restaurant_info_df: df, with filters either applied or not
    '''
    # Build the query string dynamically, excluding "any" values
    query_conditions = []
    for key, value in preferences.items():
        if value and value != "any":  # Ignore "any" values in the query
            if version == 'eq':
                query_conditions.append(f"{key} == '{value}'")
            elif version == 'ineq':
                query_conditions.append(f"{key} != '{value}'")

    query_string = ' & '.join(query_conditions)

    # Perform the query only if there's something to filter
    if query_string:
        if output == 'list':
            return restaurant_info_df.query(query_string)['restaurantname'].tolist()
        else:
            return restaurant_info_df.query(query_string)
    else:
        if output == 'list':
            return restaurant_info_df['restaurantname'].tolist()  # Return all restaurants if no query filters
        else:
            return restaurant_info_df  # Return full dataframe if no filters

# restaurant_database = pd.read_csv("data/restaurant_info.csv")
# preference_categories_dict = initiate_category_dict(restaurant_database)
# preferences_list = extract_preferences("want italian food in the south ", preference_categories_dict)
# print(query_restaurant(preferences_list, restaurant_database))
