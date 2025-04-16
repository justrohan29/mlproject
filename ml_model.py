from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def train_model(data):
    X = data[['Gender', 'Age', 'Annual Income (k$)']]
    y = data['Spending Score (1-100)']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test
