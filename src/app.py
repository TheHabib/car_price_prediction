import streamlit as st
import pandas as pd
import pickle
import numpy as np
from datetime import datetime
import os

# Streamlit UI
st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", initial_sidebar_state="collapsed")

# Define custom CSS for background color and text color
st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: black;
            color: white;
        }}
        .stMarkdown h1, h2, h3, h4, h5, h6, p {{
            color: white;
        }}
    </style>
    """, unsafe_allow_html=True
)


# Load the model and encoders
model_file = './models/car_price_prediction_model_RandomForestRegressor.pkl'
category_encoder_file = './models/category_encoder.pkl'
color_encoder_file = './models/color_encoder.pkl'
fuel_encoder_file = './models/fuel_encoder.pkl'
doors_encoder_file = './models/doors_encoder.pkl'
drive_encoder_file = './models/drive_encoder.pkl'
gear_encoder_file = './models/gear_encoder.pkl'
leather_encoder_file = './models/leather_encoder.pkl'
model_encoder_file = './models/model_encoder.pkl'
manufacturer_encoder_file = './models/manufacturer_encoder.pkl'
car_models_df = pd.read_csv("./datasets/Car_Models.csv")

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


st.title('Used Car Price Prediction App')

# Sidebar with links
st.sidebar.header("Useful Links")
st.sidebar.markdown("[GitHub Repository](https://github.com/TheHabib/car_price_prediction)")
st.sidebar.markdown("[Docker Image](https://hub.docker.com/r/your_docker_image)")


# First line: Select Manufacturer and Select Your Model
col1, col2 = st.columns(2)
with col1:
    manufacturer_name = st.selectbox('Select Manufacturer', manufacturer_encoder.classes_)
with col2:
    filtered_models = car_models_df[car_models_df['Manufacturer'] == manufacturer_name]['Model'].unique()
    model_name = st.selectbox('Select Your Model', filtered_models)

# Second line: Enter Production Year
prod_year = st.number_input('Enter Production Year', max_value=2025, value=2025, step=1, format="%d")

# Third line: Enter Engine Volume (in litres) and Enter Mileage (in km)
col3, col4 = st.columns(2)
with col3:
    engine_volume = st.number_input('Enter Engine Volume (in litres)', min_value=0.0, format="%f")
with col4:
    mileage = st.number_input('Enter Mileage (in km)', min_value=0)

# Fourth line: Select Fuel Type and Select Gear Box Type
col5, col6 = st.columns(2)
with col5:
    fuel_type = st.selectbox('Select Fuel Type', fuel_encoder.classes_)
with col6:
    gear_box_type = st.selectbox('Select Gear Box Type', gear_encoder.classes_)

# Fifth line: Select Vehicle Category and Select Number of Doors
col7, col8 = st.columns(2)
with col7:
    category = st.selectbox('Select Vehicle Category', category_encoder.classes_)
with col8:
    doors_option = st.selectbox('Select Number of Doors', doors_encoder.classes_)

# Sixth line: Does Your Car Have Leather Interior? and Select Car Color
col9, col10 = st.columns(2)
with col9:
    leather_interior = st.selectbox('Does Your Car Have Leather Interior?', ['Yes', 'No'])
with col10:
    color = st.selectbox('Select Car Color', color_encoder.classes_)

# Seventh line: Enter Number of Airbags and Enter Levy Amount
col11, col12 = st.columns(2)
with col11:
    airbags_number = st.number_input('Enter Number of Airbags:', min_value=0, value=0)
with col12:
    levy = st.number_input('Enter Levy Amount', min_value=0)

if st.button('Predict Price'):
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

    input_data.dropna(inplace=True)
    if input_data.empty:
        st.error("Invalid input: Some values were not recognized by the model.")
    else:
        prediction = model.predict(input_data)[0]
        st.success(f'Estimated Car Price: ${prediction:,.2f}')
