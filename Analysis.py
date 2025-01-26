# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the dataset
data = pd.read_csv('Housing.csv')

# Step 2: Explore the dataset
print("Dataset Preview:")
print(data.head())
print("\nDataset Information:")
print(data.info())

# Step 3: Feature Selection and Encoding
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
example_input = example_input.reindex(columns=X.columns, fill_value=0)
predicted_price = model.predict(example_input)
print(f"\nPredicted House Price for example input: {predicted_price[0]}")

# Step 9: Save the Model
import joblib
joblib.dump(model, 'house_price_model.pkl')

# Step 10: Visualization
# A. Scatter Plot: Area vs Price
plt.figure(figsize=(10, 6))
plt.scatter(data['area'], data['price'], color='blue', alpha=0.5, label='Actual Data')
plt.plot(X_test['area'], y_pred, color='red', label='Predicted Line')
plt.title('Area vs Price')
plt.xlabel('Area (sq ft)')
plt.ylabel('Price')
plt.legend()
plt.show()

# B. Residual Plot
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_test, residuals, color='purple', alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--', label='Zero Error Line')
plt.title('Residual Plot')
plt.xlabel('Actual Price')
plt.ylabel('Residuals')
plt.legend()
plt.show()

# C. Feature Importance (Coefficients)
plt.figure(figsize=(12, 8))
plt.barh(coefficients['Feature'], coefficients['Coefficient'], color='teal')
plt.title('Feature Importance')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()
