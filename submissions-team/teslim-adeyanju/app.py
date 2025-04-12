# diamond_price_app.py

# 🌟 Simple Diamond Price Prediction App
# Based on a trained regression model that expects scaled inputs

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Page settings
st.set_page_config(
    page_title='💎 Diamond Price Prediction',
    layout='centered',
    initial_sidebar_state='expanded'
)

st.title('💎 Diamond Price Predictor')

# Sidebar for input
st.sidebar.header('Enter Diamond Features')

# Categorical feature selections
cut = st.sidebar.selectbox('Cut', options=['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
color = st.sidebar.selectbox('Color', options=['J', 'I', 'H', 'G', 'F', 'E', 'D'])
clarity = st.sidebar.selectbox('Clarity', options=['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])

# Other numeric features
depth = st.sidebar.slider('Depth (%)', 43.0, 79.0, 61.0)
table = st.sidebar.slider('Table (%)', 43.0, 95.0, 57.0)
carat = st.sidebar.slider('Carat', 0.2, 5.1, 0.9)

# Instead of asking users for the PCA component, ask for the raw size measurements:
x_factor_input = st.sidebar.slider('x (Premium)', 0.0, 10.0, 5.73)
y_factor_input = st.sidebar.slider('y (Good)', 0.0, 10.0, 5.73)
z_factor_input = st.sidebar.slider('z (Very Good)', 0.0, 10.0, 3.54)

# === Compute the PCA component from raw size features ===
# Use fixed mean and std values obtained during training:
x_mean, x_std = 5.73, 1.12
y_mean, y_std = 5.73, 1.14
z_mean, z_std = 3.54, 0.71

# Standardize the raw inputs
x_std_val = (x_factor_input - x_mean) / x_std
y_std_val = (y_factor_input - y_mean) / y_std
z_std_val = (z_factor_input - z_mean) / z_std

# PCA loadings as computed during training
loading_x = 0.580088
loading_y = 0.576372
loading_z = 0.575581

# Calculate the PCA component (PCA_1)
pca_1_calculated = loading_x * x_std_val + loading_y * y_std_val + loading_z * z_std_val

# === Encode categorical features using fixed mappings (same as in training) ===
cut_map = {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4}
color_map = {'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6}
clarity_map = {'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7}

cut_enc = cut_map[cut]
color_enc = color_map[color]
clarity_enc = clarity_map[clarity]

# === Compute derived features ===
carat_squared = carat ** 2
carat_clarity = carat * clarity_enc
cut_color = cut_enc * color_enc

# === Build the final input DataFrame ===
# (The order and feature names must match what your model was trained on)
input_features = pd.DataFrame([[
    cut_enc, color_enc, clarity_enc, depth, table,
    pca_1_calculated,   # Computed PCA component from raw size inputs
    carat, carat_squared, carat_clarity, cut_color
]], columns=[
    'cut', 'color', 'clarity', 'depth', 'table', 'PCA_1',
    'carat', 'carat^2', 'carat_clarity', 'cut_color'
])

# === Load the StandardScaler, Model, and PowerTransformer ===
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('diamond_price_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('power_transformer.pkl', 'rb') as pt_file:
    pt = pickle.load(pt_file)

# Scale the input features as in training
scaled_input = scaler.transform(input_features)

# Make prediction (model was trained with Box-Cox transformed target)
pred_boxcox = model.predict(scaled_input)
pred_price = pt.inverse_transform(pred_boxcox.reshape(-1, 1))[0][0]

# Display prediction output
st.subheader('💡 Prediction')
st.write('Based on your inputs, the predicted **diamond price** is:')
st.metric(label="💵 Estimated Price (USD)", value=f"${pred_price:,.2f}")

# === Display a Historical Price Chart (Optional) ===
st.subheader("Historical Diamond Price Distribution")
try:
    # Load the historical dataset (final_dataset.pkl should be available with the Box-Cox price)
    final_dataset = pd.read_pickle('final_dataset.pkl')
    # Inverse-transform the Box-Cox target to get original price values
    original_prices = pt.inverse_transform(final_dataset['price_boxcox'].values.reshape(-1, 1))
    price_series = pd.Series(original_prices.flatten(), name="Price").sort_values().reset_index(drop=True)
    
    st.line_chart(price_series)
except Exception as e:
    st.write("Historical price distribution data not available.", e)
