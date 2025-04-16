import streamlit as st
import pandas as pd
from ml_model import train_model
import plotly.express as px


# Title and description
st.title("Customer Spending Score Predictor")
st.markdown("Input customer details to predict their Spending Score dynamically.")

# Load the dataset
@st.cache_data
def load_data():
    # Load the dataset
    data = pd.read_csv("Mall_Customers.csv")
    data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})  # Encode gender
    return data

data = load_data()

# Sidebar for user inputs
st.sidebar.header("Input Customer Details")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", min_value=18, max_value=70, value=30)
annual_income = st.sidebar.number_input("Annual Income (in $k)", min_value=0, max_value=200, value=50)

# Train the model dynamically
model, X_test, y_test = train_model(data)  # Corrected: Passing 'data', not "data"

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

# Sidebar button for detailed analysis
if st.sidebar.button("Show Dataset"):
    st.subheader("Dataset Overview")
    st.write("Here's a preview of your dataset:")
    st.dataframe(data)

# Footer message for clarity
else:
    st.sidebar.info("Click the 'Predict Spending Score' or 'Show Dataset' buttons to explore the app!")


# Add footer at the bottom of the main page
for _ in range(20):  # Add 20 blank lines to push footer down
    st.write("\n")

st.markdown('<p style="font-size:10px; text-align:center;">Made with 💖 by Rohan and teammates</p>', unsafe_allow_html=True)
