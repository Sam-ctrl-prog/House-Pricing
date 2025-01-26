# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the dataset
# Replace 'house_data.csv' with your actual dataset file
data = pd.read_csv('Housing.csv')

# Step 2: Explore the dataset
print("Dataset Preview:")
print(data.head())
print("\nDataset Information:")
print(data.info())

# Step 3: Feature Selection and Encoding
# Selecting features and target variable
features = [
    'area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom',
    'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea',
    'furnishingstatus'
]
target = 'price'

# Encoding categorical variables
categorical_features = ['mainroad', 'guestroom', 'basement', 'hotwaterheating',
                        'airconditioning', 'prefarea', 'furnishingstatus']
data = pd.get_dummies(data, columns=categorical_features, drop_first=True)

# Step 4: Split data into training and testing sets
X = data.drop(columns=[target])  # Features
y = data[target]  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# Step 7: Display Coefficients
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})
print("\nFeature Coefficients:")
print(coefficients)

# Step 8: Prediction Example
# Create a DataFrame for the example input with matching features
example_input = pd.DataFrame({
    'area': [3000],
    'bedrooms': [3],
    'bathrooms': [2],
    'stories': [2],
    'mainroad_yes': [1],
    'guestroom_yes': [0],
    'basement_yes': [0],
    'hotwaterheating_yes': [1],
    'airconditioning_yes': [1],
    'parking': [2],
    'prefarea_yes': [1],
    'furnishingstatus_semi-furnished': [0],
    'furnishingstatus_unfurnished': [0]
})

# Ensure the input aligns with the trained model's features
example_input = example_input.reindex(columns=X.columns, fill_value=0)

# Predict the house price for the example input
predicted_price = model.predict(example_input)
print(f"\nPredicted House Price for example input: {predicted_price[0]}")

# Step 9: Save the Model (Optional)
import joblib
joblib.dump(model, 'house_price_model.pkl')
