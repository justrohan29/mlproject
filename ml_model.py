import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Train the model dynamically
def train_model(data_path):
    # Load the dataset
    data = pd.read_csv(data_path)
    
    # Encode categorical variables (Gender: Male = 1, Female = 0)
    data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
    
    # Define features and target
    X = data[['Gender', 'Age', 'Annual Income (k$)']]
    y = data['Spending Score (1-100)']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a Random Forest Regressor
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    # Return the trained model and the test data
    return model, X_test, y_test