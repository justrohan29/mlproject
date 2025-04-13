import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor

# Train the model (if not already trained)
def train_model():
    # Example hardcoded data (train on your actual data beforehand)
    from sklearn.model_selection import train_test_split
    import pandas as pd
    
    data = pd.read_csv("Mall_Customers.csv")  # Replace with your actual dataset if required
    X = pd.get_dummies(data[['Gender', 'Age', 'Annual Income (k$)']], drop_first=True)
    y = data['Spending Score (1-100)']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a random forest regressor
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    # Save the trained model
    with open("spending_score_model.pkl", "wb") as f:
        pickle.dump(model, f)

# Predict the spending score based on user input
def predict_spending_score(input_data):
    # Load the pre-trained model
    with open("spending_score_model.pkl", "rb") as f:
        model = pickle.load(f)
    
    # Predict spending score
    spending_score = model.predict([input_data])
    return np.round(spending_score[0], 2)