# 2.1.0 Imported Libraries

# Standard Libraries
import os
import unittest

# Third-Party Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from scipy.stats import skew

# Project-Specific Modules
from dpputility.data_set_module import get_data_frame

# Title and Description
st.title("Diamond Price Prediction")
st.write("This app predicts the price of diamonds based on various features.")

# Load Data and Models
@st.cache_data
def load_data():
    return pd.read_csv("data/Diamonds-Data.csv")

@st.cache_resource
def load_model():
    return joblib.load("models/diamond_price_model.joblib")

data = load_data()
model = load_model()

# Sidebar for User Input
st.sidebar.header("Input Features")
carat = st.sidebar.slider("Carat", float(data["carat"].min()), float(data["carat"].max()), 0.5)
cut = st.sidebar.selectbox("Cut", data["cut"].unique())
color = st.sidebar.selectbox("Color", data["color"].unique())
clarity = st.sidebar.selectbox("Clarity", data["clarity"].unique())

# Prediction
if st.sidebar.button("Predict"):
    input_features = pd.DataFrame({
        "carat": [carat],
        "cut": [cut],
        "color": [color],
        "clarity": [clarity]
    })
    prediction = model.predict(input_features)
    st.write(f"Predicted Price: ${prediction[0]:,.2f}")