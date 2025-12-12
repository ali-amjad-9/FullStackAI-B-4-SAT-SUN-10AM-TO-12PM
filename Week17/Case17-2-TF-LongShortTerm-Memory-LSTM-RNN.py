"""
https://www.geeksforgeeks.org/long-short-term-memory-lstm-rnn-in-tensorflow/

"""

"""
1. Importing Libraries
In this step, we will import the necessary libraries like pandas, numpy, matplotlib, scikit-learn and tensorflow. Here tensorflow library is used to create the LSTM Model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

"""
2. Data Loading, Preparing and Scaling
Here we are using a dataset of monthly milk production using LSTM. You can download dataset from here.

We load the dataset of monthly milk production. The "Date" column is converted to datetime format for time series analysis.
We scale the data to a range of [0, 1] using MinMaxScaler to help the model train more effectively.
"""
data = pd.read_csv('Week17/monthly_milk_production.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
production = data['Production'].astype(float).values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(production)

"""
3. Creating Sequences and Train-Test Split
Here we generate sequences of input data and split the dataset into training and testing sets.

We use a sliding window of 12 months (1 year) of past data to predict the next month's production.
The dataset is split into training and testing sets and reshaped to match the LSTM input shape.
We split 80% data for training and 20% for testing purposes.
"""
window_size = 12
X = []
y = []
target_dates = data.index[window_size:]

for i in range(window_size, len(scaled_data)):
    X.append(scaled_data[i - window_size:i, 0])
    y.append(scaled_data[i, 0])

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
    X, y, target_dates, test_size=0.2, shuffle=False
)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
"""
4. Building the LSTM Model
This step involves defining and building the LSTM model architecture.

The model consists of two LSTM layers, each with 128 units and a dropout layer after each to prevent overfitting.
The model concludes with a Dense layer to predict a single value (next month's production).
"""
model = Sequential()
model.add(LSTM(units=128, return_sequences=True,
          input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=128))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
"""
5. Training and Evaluating the Model
In this step, we train the model on the training data and evaluate its performance.

The model is trained for 100 epochs using a batch size of 32, with 10% of the training data used for validation.
After training the model is used to make predictions on the test set and we calculate the Root Mean Squared Error (RMSE) to evaluate performance.
"""
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1)

predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions).flatten()
y_test = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()

rmse = np.sqrt(np.mean((y_test - predictions)**2))
print(f'RMSE: {rmse:.2f}')

"""
6. Visualizing Model Performance
In this step, we visualize the actual vs predicted values. A plot is generated to compare the actual milk production against the predicted values, allowing us to evaluate how well the model performs over time.
"""
plt.figure(figsize=(12, 6))
plt.plot(dates_test, y_test, label='Actual Production')
plt.plot(dates_test, predictions, label='Predicted Production')
plt.title('Actual vs Predicted Milk Production')
plt.xlabel('Date')
plt.ylabel('Production (pounds per cow)')
plt.legend()
plt.show()