# NAME: Muhammad Ali Amjad 
# SECTION A: QUESTION 1 
# Real state price prediction using machine learning

# IMPORTING  libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# now we will import machine learning models
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# checking that matrix is good or not
from sklearn.metrics import mean_squared_error, r2_score

# lets clean the data
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# STEP 1---
print("STEP 1---")

# here we will use the csv file 

filename = 'Section-A-Q1-USA-Real-Estate-Dataset-realtor-data-Rev1.csv'
data = pd.read_csv(filename)


#checking data
print("Data loaded successfully.")
print("Here are the first 5 rows:")
print(data.head(5))


# checking last 5 rows
print("\nHere are the last 5 rows:")
print(data.tail(5))

# checking data types
print("\nChecking data types (dtypes):")
print(data.dtypes)

# checking for empty values 
print("\nChecking for empty values:")
print(data.isnull().sum())

  
print("\nSummary Statistics (Describe):")
print(data.describe())
 
print("\nOriginal Data Shape (Rows, Columns):")
print(data.shape)

# STEP 2----
print("\nSTEP 2--->")

# we will access bed column
bed_column = data['bed']
print('Accessing the bed column:')
print(bed_column.head())

# now the house_size column
house_size_col = data['house_size']
print('Accessing the house_size column:')
print(house_size_col.head())

# and now multiple columns at once
print('Accessing bed, bath, and price columns:')
subset_data = data[['bed', 'bath', 'price']]
print(subset_data.head())

# now accessing specific rows using .loc
print('Accessing Row 1 using .loc:')
row1 = data.loc[1]
print(row1)


print('Accessing rows 1 to 5:')
rows_slice = data.loc[1:5]
print(rows_slice)

# Conditional things
print("Checking houses in 'New York' city:")
ny_houses = data.loc[data['city'] == 'New York']
print(ny_houses.head())

# STEP 3-----
print("\nSTEP 3----->")
# 1. filtering for city
print("filtering data for 'new york' city---")

if 'state' in data.columns and (data['state'] == 'New York').any():
    data = data[data['state'] == 'New York']
    print("Filtered for 'New York'. New Shape:", data.shape)
else:
    if 'state' in data.columns:
        unique_states = data['state'].unique()
        print("'New York' not found in data. Available states:", unique_states)
    else:
        print("No 'state' column found; skipping state filter.")
    print("Skipping state filter; using full dataset. Shape:", data.shape)

# 2. Removing Duplicates
print("Removing duplicate rows---")
data.drop_duplicates(inplace=True)
print("Shaping after removing duplicated rows:", data.shape)
# 3. Handling values of missing prices
print("Dropping rows where Price is null---")
data.dropna(subset=['price'], inplace=True)

# 4. Removing Outliers
# IQR method
print("Calculating IQR---")

price_col = data['price']
Q1 = price_col.quantile(0.25)
Q3 = price_col.quantile(0.75)
IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

print(f"Lower Price Limit: {lower_limit}")
print(f"Upper Price Limit: {upper_limit}")

# lets filter the data
data = data[(data['price'] >= lower_limit) & (data['price'] <= upper_limit)]
print("Shape after removing outliers:", data.shape)

# STEP 4----

print("\nSTEP 4:")

# Graph 1: now we will make a bar graph of distribution of prices
plt.figure(figsize=(10, 6))
plt.hist(data['price'], bins=50, color='skyblue', edgecolor='black')
plt.title('Histogram of House Prices (New York)')
plt.xlabel('Price ($)')
plt.ylabel('Count')
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('graph1_price_histogram.png')
print("Graph 1 (Histogram) saved successfully.")
plt.show()

# Graph 2: now a graph for house size vs price
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['house_size'], y=data['price'], alpha=0.5, color='green')
plt.title('House Size vs Price')
plt.xlabel('House Size (sq ft)')
plt.ylabel('Price ($)')
plt.grid(True)
plt.savefig('graph2_size_vs_price.png')
print("Graph 2 (Scatter Plot) saved successfully.")
plt.show()

# STEP 5----

print("\nSTEP 5:")

# 1. House Age
# lets convert column to datetime
data['prev_sold_date'] = pd.to_datetime(data['prev_sold_date'])

data['sold_year'] = data['prev_sold_date'].dt.year

# 2. now we will convert bed to bath ratio
data['bed_bath_ratio'] = data['bed'] / (data['bath'] + 0.001)

print("Added 'sold_year' and 'bed_bath_ratio'.")
print(data[['sold_year', 'bed_bath_ratio']].head())

# STEP 6----

print("\nSTEP 6:")

# Selecting the features (X) and the target (y)
# I am dropping columns that are text (like street) or not useful anymore
drop_cols = ['price', 'prev_sold_date', 'status', 'street', 'brokered_by']
X = data.drop(drop_cols, axis=1)
y = data['price']

print("Features in X:", X.columns.tolist())
print("Target is Price.")

# Ensure X and y have aligned integer indices and matching lengths
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)
print("Sanity check - shapes after reset:", X.shape, y.shape)
if len(X) != len(y):
    raise ValueError(f"Mismatched samples: len(X)={len(X)} vs len(y)={len(y)}")

# Splitting the data into Training set and Test set
# I am using 80% for training and 20% for testing
# random_state=42 makes sure the split is the same every time
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training Data Size: {X_train.shape}")
print(f"Testing Data Size: {X_test.shape}")

# STEP 7: SETTING UP THE PIPELINE
 
print("\n STEP 7: SETTING UP TRANSFORMERS")

# We need to handle missing values and scale the numbers.
# I am using a Pipeline because it is cleaner.

# Numerical columns (bed, bath, size, etc.)
num_features = ['bed', 'bath', 'acre_lot', 'house_size', 'sold_year', 'bed_bath_ratio']

# Categorical columns (city, zip code)
cat_features = ['city', 'zip_code']

# Pipeline for Numerical Data
# 1. Fill missing values with the Median
# 2. Scale the numbers using Standard Scaler
num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Pipeline for Categorical Data
# 1. Fill missing values with 'most_frequent'
# 2. Turn text into numbers using OneHotEncoder
cat_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', max_categories=20))
])

# Combining both pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features)
    ])

print("Preprocessor is ready.")


# STEP 8: TRAINING MODELS


print("\n STEP 8: TRAINING 3 MODELS ")

# I need to compare 3 models for the assignment.

# LINEAR REGRESSION
print("Training Model 1: Linear Regression...")
model_lr = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# fit the model
model_lr.fit(X_train, y_train)

# predict
y_pred_lr = model_lr.predict(X_test)

# calculate error
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
print(f"Linear Regression RMSE: {rmse_lr}")


# MODEL 2: RANDOM FOREST 
print("\nTraining Model 2: Random Forest (this might take a minute)...")
model_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=50, random_state=42))
])

# fit the model
model_rf.fit(X_train, y_train)

# predict
y_pred_rf = model_rf.predict(X_test)

# calculate error
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
print(f"Random Forest RMSE: {rmse_rf}")


# MODEL 3: XGBOOST 
print("\nTraining Model 3: XGBoost...")
model_xgb = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(n_estimators=100, learning_rate=0.1, n_jobs=-1))
])

# fit the model
model_xgb.fit(X_train, y_train)

# predict
y_pred_xgb = model_xgb.predict(X_test)

# calculating errors
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
print(f"XGBoost RMSE: {rmse_xgb}")

# STEP 9: FINAL COMPARISON AND PLOT


print("\n STEP 9: RESULTS ")

# Comparing the errors
print("Final Comparison of Models (RMSE - Lower is Better):")
print(f"1. Linear Regression: {int(rmse_lr)}")
print(f"2. Random Forest:     {int(rmse_rf)}")
print(f"3. XGBoost:           {int(rmse_xgb)}")

# Calculating R2 Score for the best model (usually XGBoost or Random Forest)
best_model_r2 = r2_score(y_test, y_pred_xgb)
print(f"R2 Score of Best Model: {best_model_r2:.3f}")

# Final Graph: Actual vs Predicted Prices
print("\nCreating Final Prediction Graph...")

plt.figure(figsize=(12, 6))

# I am plotting the first 100 points so the graph is easy to read
# Plotting the Real Prices
plt.plot(y_test.values[:100], color='blue', label='Actual Price', marker='o')

# Plotting the Predicted Prices (from XGBoost)
plt.plot(y_pred_xgb[:100], color='red', label='Predicted Price', linestyle='--')

plt.title('Actual vs Predicted Prices (First 100 Houses)')
plt.xlabel('House Index')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True)

plt.show()
