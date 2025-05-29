import streamlit as st
import pandas as pd
from ml_model import train_model
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_log_error, median_absolute_error

# Title and description
st.title("Customer Spending Score Predictor")
st.markdown("Input customer details to predict their Spending Score dynamically.")

# Load the dataset
@st.cache_data
def load_data():
    # Load the dataset
    try:
        data = pd.read_csv("Mall_Customers.csv")
        data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})  # Encode gender
        return data
    except FileNotFoundError:
        st.error("Dataset file 'Mall_Customers.csv' not found. Please ensure it is in the correct directory.")
        return pd.DataFrame()

data = load_data()

# Sidebar for user inputs
st.sidebar.header("Input Customer Details")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", min_value=18, max_value=70, value=30)
annual_income = st.sidebar.number_input("Annual Income (in $k)", min_value=0, max_value=200, value=50)

# Train the model dynamically
model, X_test, y_test = train_model(data)  # Passing the loaded data

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

# Sidebar button for model accuracy metrics
if st.sidebar.button("Show Model Accuracy"):
    st.subheader("Model Accuracy and Metrics")
    # Get predictions for test data
    test_predictions = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, test_predictions)
    medae = median_absolute_error(y_test, test_predictions)

    # Display metrics
    st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
    st.write(f"**Median Absolute Error (MedAE):** {medae:.2f}")
    st.info("These metrics evaluate the model's accuracy on test data.")

# Sidebar button for visual charts
if st.sidebar.button("Show Visual Charts"):
    st.subheader("Visualizations")
    st.write("Explore the distribution and correlations within the dataset:")

    # Gender Distribution Pie Chart
    st.markdown("### Gender Distribution")
    gender_counts = data['Gender'].value_counts()
    fig = px.pie(values=gender_counts.values, names=['Male', 'Female'], title="Gender Distribution")
    st.plotly_chart(fig)

    # Age Distribution Histogram
    st.markdown("### Age Distribution")
    fig = px.histogram(data, x="Age", nbins=15, title="Age Distribution", color_discrete_sequence=["blue"])
    st.plotly_chart(fig)

    # Annual Income vs Spending Score Scatter Plot
    st.markdown("### Annual Income vs Spending Score")
    fig = px.scatter(data, x="Annual Income (k$)", y="Spending Score (1-100)", color=data['Age'],
                     title="Annual Income vs Spending Score",
                     labels={"Annual Income (k$)": "Annual Income", "Spending Score (1-100)": "Spending Score"})
    st.plotly_chart(fig)

    # Correlation Matrix Heatmap
    st.markdown("### Correlation Matrix")
    corr_matrix = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].corr()
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        title="Correlation Matrix Heatmap",
        color_continuous_scale='plasma'  # Updated colorscale
    )
    st.plotly_chart(fig)

# Footer message for clarity
st.sidebar.markdown("Made By:")
st.sidebar.markdown("👩‍💻 Rohan")
