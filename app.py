import streamlit as st
import pandas as pd
import pickle
import numpy as np
from lightgbm import LGBMRegressor

# Streamlit interface
st.title('Insurance Premium Prediction')

# Load the model and features
with open('lgbm_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('top_10_features.pkl', 'rb') as file:
    top_features = pickle.load(file)

# Define the function to predict
def predict_premium(user_data):
    # Ensure categorical columns are treated correctly
    for col in top_features:
        if user_data[col].dtype == 'object':
            user_data[col] = user_data[col].astype('category')
    
    # Predict using the model
    prediction = model.predict(user_data[top_features])
    return prediction[0]

# Collect user input in the sidebar
st.sidebar.header('User Input')
input_data = {}
for feature in top_features:
    input_data[feature] = st.sidebar.text_input(f"Enter value for {feature}")

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


