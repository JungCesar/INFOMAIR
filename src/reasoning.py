import random
import csv
import pandas as pd
import os

'''
def update_restaurant_info(added_features_dict, restaurant_info_path):
    df = pd.read_csv(restaurant_info_path)
    for feature in added_features_dict.keys():
        df[feature] = df.apply(lambda x: random.choice(added_features_dict[feature]), axis=1)
    # update_name_split = restaurant_info_path.split('/')
        
    directory = os.path.dirname(restaurant_info_path)
    df.to_csv(directory + '/updated_restaurant_info.csv', index=False)


added_features = {
    "quality" : ["bad food", "good food"],
    "crowdedness": ["busy", "not busy"],
    "length_of_stay": ["short stay", "long stay"]
}

<<<<<<< HEAD
update_restaurant_info(added_features, '../data/restaurant_info.csv')
=======
update_restaurant_info(added_features, 'data/restaurant_info.csv')
'''
>>>>>>> 3fca223b2d207f8524646fc7a53ff9daeab88b9d

# additional_preferences = {
#  'romantic' : ['romantic'],
#  'assigned seats' : ['assigned seats'],
#  'children' : ['children'],
#  'touristic' : ['touristic']
# }

# selected_added_pref = extract_preferences('i would like a romantic restaurant but touristic too', additional_preferences)
# print(selected_added_pref)


def inference_rules(add_preferences):
#pricerange,area,food,quality,crowdedness,length_of_stay
    filters_true={}
    filters_false={}
    reason = ''
    if add_preferences["touristic"]:
        filters_true['quality'] = 'good food'
        filters_true['pricerange']= 'cheap'
        filters_false['food'] = 'romanian'
        reason += "It is touristic because it has good quality food, the food is cheap and the food is not romanian. "
    if add_preferences['assigned seats']:
        filters_true['crowdedness']='busy'
        reason += "It has assigned seats as it is busy. "
    if add_preferences['children']:
        filters_true['length_of_stay'] = 'short stay'
        reason+= "It is ideal for children because it is ideal for short stay. "
    if add_preferences['romantic']:
        filters_true['length_of_stay'] = 'long stay'
        filters_false['crowdedness'] = 'busy'
        reason += "It is romantic because it is ideal for long stay and it is not busy. "    
    return filters_true, filters_false, reason

<<<<<<< HEAD
filters_true, filters_false= inference_rules(selected_added_pref)


restaurant_df=pd.read_csv('../data/updated_restaurant_info.csv')
# print(restaurant_df.columns)
restaurant_df = query_restaurant(filters_true, restaurant_df, output = 'df', version ='eq')
print(query_restaurant(filters_false, restaurant_df, output = 'list', version ='ineq'))
=======
# filters_true, filters_false= inference_rules(selected_added_pref)
# restaurant_df=pd.read_csv('data/updated_restaurant_info.csv')
# # print(restaurant_df.columns)
# restaurant_df = query_restaurant(filters_true, restaurant_df, output = 'df', version ='eq')
# print(query_restaurant(filters_false, restaurant_df, output = 'list', version ='ineq'))
>>>>>>> 3fca223b2d207f8524646fc7a53ff9daeab88b9d
