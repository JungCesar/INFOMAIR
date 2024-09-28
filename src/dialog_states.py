# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 09:32:50 2024

@author: lucmi
"""
info = {"food": [],
               "pricerange": [], 
               "area": []}

import keyword_mapping as km
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import joblib as jl
model = jl.load("models/logistic_regression_classifier.joblib")
vectorizer = jl.load("models/logistic_regression_classifier_vectorizer.joblib")
restaurant_database = pd.read_csv("../data/restaurant_info.csv")
preference_categories_dict = km.initiate_category_dict(restaurant_database)
#ex = "hello"
#tex = vectorizer.transform([ex])
#print(model.predict(tex))
def start(model):
    print("hello, welcome to restaurant recommender")
    text = input()
    input1 = vectorizer.transform([text])
    label = model.predict(input1)
           
    if label == "inform":
        what = km.extract_preferences(text, preference_categories_dict)
        if len(what["food"]) != None:
               info["food"].append(what["food"])
               print("You asked for ", what["food"])

        if len(what["pricerange"]) != None:
               info["pricerange"].append(what["pricerange"])
               print("You asked for ", what["pricerange"])

        if len(what["area"]) != None:
               info["area"].append(what["area"])
               print("You asked for ", what["area"])

        for information in info.keys():
            if len(info[information]) == 0:
                if information == "food":
                    askfood()
                if information == "pricerange":
                    askprice()
                if information == "area":
                    asklocation()
    else:
        askfood()
        askprice()
        asklocation()
    input2 = f'want {info["food"]} food that is {info["pricerange"]} in the {info["area"]}'
    preferences_list = km.extract_preferences(input2, preference_categories_dict)
    print(km.get_restaurant(preferences_list, restaurant_database))


def askfood():
    print("What type of food do you want?")
    text = input()
    input1 = vectorizer.transform([text])
    label = model.predict(input1)
    if label == "inform":
        what = km.extract_preferences(text, preference_categories_dict)
        if what["food"] != None:
               info["food"].append(what["food"])
               print("You asked for ", what["food"])
        else:
            print("answer the question")
            askfood()
        
    else:
        print("You should answer the question:")
        askfood()
        
def askprice():
    print("What type of pricerange do you want?")
    text = input()
    input1 = vectorizer.transform([text])
    label = model.predict(input1)
    if label == "inform":
        what = km.extract_preferences(text, preference_categories_dict)
        if what["pricerange"] != None:
               info["pricerange"].append(what["pricerange"])
               print("You asked for ", what["pricerange"])
        else:
            print("answer the question")
            askprice()
    else:
        print("You should answer the question:")
        askprice()
def asklocation():
    print("What type of location do you want?")
    text = input()
    input1 = vectorizer.transform([text])
    label = model.predict(input1)
    if label == "inform":
        what = km.extract_preferences(text, preference_categories_dict)
        if what["area"] != None:
               info["area"].append(what["area"])
               print("You asked for ", what["area"])
        else:
            print("answer the question")
            asklocation()
    else:
        print("You should answer the question:")
        asklocation()
start(model)