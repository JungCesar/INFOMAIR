#dialogmanament
import pandas as pd
"""je moeder"""
info = []
food = []
area = []
pricerange = []

def dataread(filename):
    df = pd.read_csv(filename)
    #print(df)
    food = []
    area = []
    pricerange = []
    for i in df.iloc:
        #print(i[2])
        food.append(i[3])
        area.append(i[2])
        pricerange.append(i[1])
    food = set(food)
    area = set(area)
    pricerange = set(pricerange) 
    return food, area, pricerange

def match(input1):
    for word in input1:
        if word in food:
            return "food"
        if word in pricerange:
            return "pricerange"
        if word in area:
            return "area"
        else:
            return "somethingelse"
            
def start(input1):
    matched = match(input1)
    if matched == "food":
        #food
    if matched == "price":
        #price
    if matched == "area":
        #food
    else:
        print("are you looking for your fatass mom restaurant?")
        #continue
print("yeao")
food, area, pricerange = dataread("../data/restaurant_info.csv")
print("system: Welcome to the best restaurant chatbot ever. What kind of food do you like?")
#input1 = input()
print(pricerange)
#start(input1)
