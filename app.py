import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Title and description
st.title("Customer Spending Score Predictor")
st.markdown("Input customer details to predict their Spending Score.")

# Load the dataset (no need to upload; it's predefined)
@st.cache_data
def load_data():
    # Example dataset provided directly within the script
    data = pd.read_csv("Mall_Customers.csv")
    data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})  # Encode gender
    return data

# Train the model dynamically
@st.cache_data
def train_model(data):
    X = data[['Gender', 'Age', 'Annual Income (k$)']]
    y = data['Spending Score (1-100)']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test

# Load and prepare the data
data = load_data()
model, X_test, y_test = train_model(data)

# Sidebar for user inputs
st.sidebar.header("Input Customer Details")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", min_value=18, max_value=70, value=30)
annual_income = st.sidebar.number_input("Annual Income (in $k)", min_value=0, max_value=200, value=50)

# Prepare user input for prediction
user_data = pd.DataFrame({
    'Gender': [1 if gender == "Male" else 0],
    'Age': [age],
    'Annual Income (k$)': [annual_income]
})

# Predict Spending Score
if st.sidebar.button("Predict Spending Score"):
    spending_score = model.predict(user_data)[0]
    st.subheader("Prediction Result")
    st.write(f"**Predicted Spending Score:** {round(spending_score, 2)}")

    # Recommendations based on spending score
    if spending_score < 40:
        st.warning("Low Spending Score. Consider budget-friendly products.")
    elif 40 <= spending_score < 70:
        st.info("Moderate Spending Score. Customers may prefer mid-range products.")
    else:
        st.success("High Spending Score. Great candidates for premium products!")

