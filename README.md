# PRODIGY_ML_Task_1
**Task:** Implement a linear regression model to predict the price of houses based on their square footage, number od bathrooms and bedrooms.

# Import Libraries
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, r2_score 
import matplotlib.pyplot as plt 
import seaborn as sns

# Load the dataset
data = pd.read_csv('house_prices.csv')

#Display the first few rows of the dataset 
print(data.head())

# Preprocess the Data
#Check for any missing values and handle them appropriately. 
print(data.isnull().sum())

#Drop rows with missing values
data = data.dropna()

# Define Features and Target Variable
#Select the features (sqft_living, bathrooms, and bedrooms) and the target variable (price). 
#Define features and target variable 
X = data[['square_footage', 'num_bathrooms', 'num_bedrooms']] 
y = data['price']

# Split the Data
#Split the data into training and testing sets. 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Model
#Create the linear regression model 
model = LinearRegression()

#Train the model 
model.fit(X_train, y_train)

# Make Predictions
#Make predictions on the test set 
y_pred = model.predict(X_test)

# Evaluate the Model
#Calculate Mean Squared Error 
mse = mean_squared_error(y_test, y_pred) 
print(f'Mean Squared Error: {mse}')

#Calculate R-squared 
r2 = r2_score(y_test, y_pred) 
print(f'R-squared: {r2}')

# Visualize the Results
#Plot the actual vs predicted prices 
plt.figure(figsize=(10, 6)) 
plt.scatter(y_test, y_pred, alpha=0.7) 
plt.xlabel('Actual Prices') 
plt.ylabel('Predicted Prices') 
plt.title('Actual vs Predicted Prices') 
plt.show()
