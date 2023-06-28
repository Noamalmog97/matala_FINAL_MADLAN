#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 18:32:02 2023

@author: noam
"""

from flask import Flask, render_template, request
from sklearn.preprocessing import OneHotEncoder

import pandas as pd
import pickle
import joblib
from madlan_data_prep import cleanin_data

app = Flask(__name__)
# Load the trained model
model = joblib.load('trained_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the form
    data = request.form

    City = str(data['City'])
    type1 = str(data['type'])
    condition = str(data['condition'])
    Area = int(data['Area'])
    hasElevator = int(data.get('hasElevator', 0))
    hasParking = int(data.get('hasParking', 0))
    hasMamad = int(data.get('hasMamad ', 0))
    hasBalcony = int(data.get('hasBalcony', 0))
    # Create a feature array
    print(hasElevator)
    data = {'City': City, 'type': type1, 'condition': condition,'Area': Area,
            'hasElevator': hasElevator, 'hasParking': hasParking,
              'hasMamad ': hasMamad,'hasBalcony': hasBalcony
            }
  
    df = pd.DataFrame(data, index=[0])
    # Make a prediction
    y_pred =round( model.predict(df)[0])

    # Return the predicted price
    return render_template('index.html', price=y_pred)

if __name__ == '__main__':
    app.run()
