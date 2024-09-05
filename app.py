import streamlit as st
import pandas as pd
import pickle
import numpy as np
from lightgbm import LGBMRegressor
# Apply custom theme using CSS
st.markdown("""
    <style>
    /* Custom colors */
    .css-1d391kg { /* Button styling */
        background-color: #003366; /* Navy Blue */
        color: #FFFFFF; /* White text */
    }

    .css-1dp5N4N { /* Header styling */
        color: #003366; /* Navy Blue text */
    }

    /* Background colors */
    .css-1v0mbdj { /* Main background */
        background-color: #FFFFFF; /* White */
    }
    .css-1i5b9e8 { /* Sidebar background */
        background-color: #F0F2F6; /* Light Grey */
    }

    /* Text color */
    body {
        color: #000000; /* Black text */
        font-family: 'Arial', sans-serif; /* Default font */
    }

    /* Custom font from Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@400;700&display=swap');
    body {
        font-family: 'Merriweather', serif;
    }
    </style>
    """, unsafe_allow_html=True)

# set the page title and favicon
st.set_page_config(page_title="Insurance Premium Prediction", page_icon="Athena.png")
# adding logo to web-app
st.image("Athena.png", width=100)
# Streamlit interface
st.title('Insurance Premium Prediction')

# Load the model
with open('lgbm_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the top features without latitude and longitude
with open('top_features.pkl', 'rb') as file:
    top_features = pickle.load(file)

# Define the categories for "Construction Type"
construction_types = ['Wood', 'Brick', 'Concrete']

# Function to predict
def predict_premium(user_data):
    # Ensure categorical columns are treated correctly
    for col in top_features:
        if user_data[col].dtype == 'object':
            user_data[col] = user_data[col].astype('category')
    
    # Predict using the model
    prediction = model.predict(user_data[top_features])
    return prediction[0]

# Collect user input in the sidebar
input_data = {}
for feature in top_features:
    if feature == 'Construction Type':
        input_data[feature] = st.sidebar.selectbox(f"Select {feature}", construction_types)
    else:
        input_data[feature] = st.sidebar.text_input(f"{feature}")

# Convert input data to DataFrame
user_df = pd.DataFrame([input_data])

# Predict when button is pressed
if st.sidebar.button('Predict Premium'):
    try:
        user_df = user_df.apply(pd.to_numeric, errors='ignore')  # Convert inputs to numeric if necessary
        premium = predict_premium(user_df)
        st.write(f"Predicted Insurance Premium: ${premium:.2f}")
    except Exception as e:
        st.error(f"Error: {e}")
