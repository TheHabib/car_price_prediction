import streamlit as st
import pandas as pd
import pickle
import random
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from datetime import datetime


# Load the model and encoders
model_file = '../models/car_price_prediction_model_RandomForestRegressor.pkl'
category_encoder_file = '../models/category_encoder.pkl'
color_encoder_file = '../models/color_encoder.pkl'
fuel_encoder_file = '../models/fuel_encoder.pkl'
doors_encoder_file = '../models/doors_encoder.pkl'
drive_encoder_file = '../models/drive_encoder.pkl'
gear_encoder_file = '../models/gear_encoder.pkl'
leather_encoder_file = '../models/leather_encoder.pkl'
model_encoder_file = '../models/model_encoder.pkl'
manufacturer_encoder_file = '../models/manufacturer_encoder.pkl'

# Load pickled objects
def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

model = load_pickle(model_file)
category_encoder = load_pickle(category_encoder_file)
color_encoder = load_pickle(color_encoder_file)
fuel_encoder = load_pickle(fuel_encoder_file)
doors_encoder = load_pickle(doors_encoder_file)
drive_encoder = load_pickle(drive_encoder_file)
gear_encoder = load_pickle(gear_encoder_file)
leather_encoder = load_pickle(leather_encoder_file)
model_encoder = load_pickle(model_encoder_file)
manufacturer_encoder = load_pickle(manufacturer_encoder_file)

# Define preprocessing functions
def process_mileage(x):
    return int(x)

def process_engine_volume(x):
    return float(x)

def process_levy(x):
    return int(x) if x != '-' else 0

def process_age(x):
    return datetime.now().year - x

def process_leather(x):
    return 1.0 if x == 'Yes' else 0.0

def process_airbags(x):
    return int(x)

def encode_value(encoder, value, column_name):
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    else:
        st.error(f"Invalid input: '{value}' not recognized in column '{column_name}'")
        return np.nan

# Streamlit UI
st.set_page_config(page_title="Car Price Prediction", page_icon="🚗")  # Set browser tab title

st.title('Used Car Price Prediction App')

st.sidebar.header('User Input Parameters')





manufacturer_name = st.sidebar.selectbox('Select Manufacturer', manufacturer_encoder.classes_)
model_name = st.sidebar.selectbox('Select Your Model', model_encoder.classes_)
prod_year = st.sidebar.number_input('Enter Production Year', max_value=2025, value=2025, step=1, format="%d")
engine_volume = st.sidebar.number_input('Enter Engine Volume (in litres)', min_value=0.0, format="%f")
mileage = st.sidebar.number_input('Enter Mileage (in km)', min_value=0)
fuel_type = st.sidebar.selectbox('Select Fuel Type', fuel_encoder.classes_)
gear_box_type = st.sidebar.selectbox('Select Gear Box Type', gear_encoder.classes_)
category = st.sidebar.selectbox('Select Vehicle Category', category_encoder.classes_)
doors_option = st.sidebar.selectbox('Select Number of Doors', doors_encoder.classes_)
leather_interior = st.sidebar.selectbox('Does Your Car Have Leather Interior?', ['Yes', 'No'])
color = st.sidebar.selectbox('Select Car Color', color_encoder.classes_)
airbags_number = st.sidebar.number_input('Enter Number of Airbags:', min_value=0, value=0)
levy = st.sidebar.number_input('Enter Levy Amount', min_value=0)


#model_name = st.sidebar.text_input('Enter Model Name')
#manufacturer_name = st.sidebar.text_input('Enter Manufacturer Name')









if st.sidebar.button('Predict Price'):
    # Preprocess input data
    input_data = pd.DataFrame({
        'Levy': [process_levy(levy)],
        'Manufacturer': [encode_value(manufacturer_encoder, manufacturer_name, 'Manufacturer')],
        'Model': [encode_value(model_encoder, model_name, 'Model')],
        'Category': [encode_value(category_encoder, category, 'Category')],
        'Leather interior': [process_leather(leather_interior)],
        'Fuel type': [encode_value(fuel_encoder, fuel_type, 'Fuel type')],
        'Engine volume': [process_engine_volume(engine_volume)],
        'Mileage': [process_mileage(mileage)],
        'Gear box type': [encode_value(gear_encoder, gear_box_type, 'Gear box type')],
        'Doors': [encode_value(doors_encoder, doors_option, 'Doors')],
        'Color': [encode_value(color_encoder, color, 'Color')],
        'Airbags': [process_airbags(airbags_number)],
        'Age': [process_age(prod_year)]        
    })

    input_data.dropna(inplace=True)  # Drop rows with NaN values
    if input_data.empty:
        st.error("Invalid input: Some values were not recognized by the model.")
    else:
        
        prediction = model.predict(input_data)[0]
        st.success(f'Estimated Car Price: ${prediction:,.2f}')
