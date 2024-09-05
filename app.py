import streamlit as st
import pandas as pd
import pickle
import numpy as np
from lightgbm import LGBMRegressor
# adding logo to web-app
# Center the image using HTML
st.markdown(
    """
    <div style='display: flex; justify-content: center;'>
        <img src='PRICE_PREDICTION_MODEL/Athena.png' width='150'/>
    </div>
    """,
    unsafe_allow_html=True
)
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
