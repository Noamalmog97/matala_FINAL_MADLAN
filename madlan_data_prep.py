#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 17:14:55 2023

@author: noam
"""

import pandas as pd
import re
import numpy as np
import seaborn as sns
import os
from datetime import datetime

path='output_all_students_Train_v10.xlsx'

def cleanin_data(path):
    data_madlan = pd.read_excel(path)
    data_madlan.dropna(subset = ['price'], inplace = True)


    data_madlan["price"] = data_madlan["price"].astype(str)  
    data_madlan["price"] = data_madlan["price"].apply(lambda x: re.sub(r'\D', '', x) if x else '') 
    data_madlan["price"] = pd.to_numeric(data_madlan["price"])
    data_madlan.dropna(subset = ['price'], inplace = True)
    
    data_madlan["Area"] = data_madlan["Area"].astype(str)  
    data_madlan["Area"] = data_madlan["Area"].apply(lambda x: re.sub(r'\D', '', x) if x else '') 
    data_madlan["Area"] = pd.to_numeric(data_madlan["Area"])
    
    text_columns = ['Street', 'city_area','description ']
    data_madlan[text_columns] = data_madlan[text_columns].astype(str)  
    data_madlan[text_columns] = data_madlan[text_columns].apply(lambda x: x.str.replace(r'[^\w\s]', ''))
        
    #Adding columns: ״floor״ and the number of floors that currently exist "total_floors"
    data_madlan['floor'] = data_madlan['floor_out_of'].str.extract(r'קומה\s(\d+)')
    data_madlan['floor'] = data_madlan['floor'].fillna(0).astype(int)

    data_madlan['total_floors'] = data_madlan['floor_out_of'].str.extract(r'מתוך\s(\d+)')
    data_madlan['total_floors'] = data_madlan['total_floors'].fillna(0).astype(int)    
    
    #entrance_date's column 
    data_madlan['entranceDate '] = data_madlan['entranceDate '].replace('גמיש', 'flexible')
    data_madlan['entranceDate '] = data_madlan['entranceDate '].replace('לא צויין', 'not_defined')
    data_madlan['entranceDate '] = data_madlan['entranceDate '].replace('מיידי', 'Less_than_6_months')
    
    to_date = pd.to_datetime(data_madlan['entranceDate '], errors='coerce').notna()
    today = pd.to_datetime(datetime.now().date())
    data_madlan.loc[to_date, 'time_difference'] = (today - pd.to_datetime(data_madlan.loc[to_date, 'entranceDate '])).dt.days / 30
    bins = [-float('inf'), 6, 12, float('inf')]
    labels = ['Less_than_6_months', 'months_6_12', 'Above_year']
    data_madlan.loc[to_date, 'entranceDate '] = pd.cut(data_madlan.loc[to_date, 'time_difference'], bins=bins, labels=labels)
    data_madlan['entranceDate '] = data_madlan['entranceDate '].fillna('invalid_value')
    data_madlan = data_madlan.drop(['time_difference'], axis=1)
    
    # boolean fields as 0 or 1 
    boolean_columns = ['hasElevator ', 'hasParking ', 'hasBars ', 'hasStorage ', 'hasAirCondition ', 'hasBalcony ', 'hasMamad ', 'handicapFriendly ']
    data_madlan[boolean_columns].fillna(0, inplace = False )
    data_madlan[boolean_columns] = data_madlan[boolean_columns].astype(str)
    replace_to_dict = {'יש': 1, 'יש ממ״ד': 1, 'יש מרפסת': 1, 'יש מיזוג אוויר': 1,'יש מיזוג אויר': 1, 'נגיש לנכים': 1,
                        'נגיש': 1,"לא נגיש":0 ,'yes': 1, 'TRUE': 1, 'True': 1, 'יש מחסן': 1, 'יש סורגים': 1,
                        'יש חנייה': 1,'יש חניה': 1, 'יש מעלית': 1, 'אין': 0, 'לא': 0, 'אין חניה': 0,
                        'אין ממ״ד': 0, 'אין מרפסת': 0, 'אין מחסן': 0, 'אין סורגים': 0,
                        'אין מעלית': 0, 'אין מיזוג אויר': 0, 'לא נגיש לנכים': 0, 'no': 0,'לא':0,
                        'FALSE': 0, 'False': 0,'כן':1, 'יש ממ״ד':1, 'יש ממ"ד':1, 'אין ממ"ד':0,"nan":0}
    data_madlan[boolean_columns] = data_madlan[boolean_columns].replace(replace_to_dict)
    
    #room number column
    data_madlan["room_number"]=data_madlan["room_number"].apply(lambda x: str(x))
    data_madlan["room_number"]=data_madlan["room_number"].apply(lambda x: re.sub(r"[^0-9.]", "", x))
    data_madlan["room_number"]=data_madlan["room_number"].apply(lambda x: float(x)  if x != '' else None)
    data_madlan["floor"]=data_madlan["room_number"].astype(float)
    
    #fixed columns "area" and "room_number" and data for the model with relevant column
    data_for_model=data_madlan[["City","Area",'city_area',"type","room_number","furniture ","condition ","entranceDate ","hasElevator ", 'hasParking ', 'hasStorage ', 'hasAirCondition ', 'hasBalcony ', 'hasMamad ', 'handicapFriendly ',"price"]]
    data_for_model['Area'] = data_for_model['Area'].fillna(data_for_model['Area'].mean())
    data_for_model['room_number'] = data_for_model['room_number'].fillna(data_madlan['room_number'].mean())
   
    
    #column for the model ONE HOT ENCODING 
    column_for_encode= ["City",'city_area']
    encoded_df= pd.get_dummies(data_for_model[column_for_encode],prefix=column_for_encode, prefix_sep='_', dtype=int,dummy_na=True)
    
    #another column for the model 
    encoded_df["Area"]=data_for_model["Area"].values
    encoded_df["hasElevator "]=data_for_model["hasElevator "].values
    encoded_df['hasMamad ']=data_for_model['hasMamad '].values
    encoded_df["hasBalcony "]=data_for_model["hasBalcony "].values
    encoded_df.columns
    
    #X and Y
    y=data_for_model["price"].values
    x=encoded_df.values
    
    return encoded_df,x,y


encoded_df,x,y= cleanin_data(path)