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


def extract_preferences(inform_text, preference_categories_dict):
    #text that is classified as inform will be the input
    #preference_categories_dict: key: (str)feature, value: list of possible values
    #the output is a dictionary in the format of 'preferences' below, with the found preferences per category 
    preferences = {category: None for category in preference_categories_dict.keys()}
    
    inform_text = inform_text.split()

    for category, keywords in preference_categories_dict.items(): 
        for word in inform_text: #iterates every category and sees if a match is found on the keywords 
            # (ask if we can vectorize the words to find synonyms or just settle with edit distance)
            # print(word, keywords)
            closest_match = match_edit_dist(word, keywords, 1)
            if closest_match:
                preferences[category] = closest_match
                # print('closest match '+closest_match+ ' in category ' + category)
                break  
    return preferences



def query_restaurant(preferences, resaurant_info_df, output = 'list', version ='eq'):

    '''
    lookup to the database for the criteria extracted
    inputs:
    preferenfes: dict, key: column name, value: the value for that feature in the df
    restaurant_info_df: df, with filters either applied or not
    '''
    if version == 'eq': #equality filter
        query_string = ' & '.join([f"{key} == '{value}'" for key, value in preferences.items() if value])
    else: #inequality filter
        query_string = ' & '.join([f"{key} != '{value}'" for key, value in preferences.items() if value])

    if query_string:
        # print(query_string)
        if output=='list':
            return (resaurant_info_df.query(query_string))['restaurantname'].tolist()
        else:
            return resaurant_info_df.query(query_string)            
    else:
        if output=='list':
            return resaurant_info_df['restaurantname'].tolist()


restaurant_database = pd.read_csv("data/restaurant_info.csv")
preference_categories_dict = initiate_category_dict(restaurant_database)
preferences_list = extract_preferences("want italian food in the south ", preference_categories_dict)
print(query_restaurant(preferences_list, restaurant_database))