import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.optimizers import Adam
from keras.metrics import Precision, Recall


"""
2. Loading the Dataset
The dataset we’re using is a time-series dataset containing daily temperature data i.e forecasting dataset. It spans 8,000 days starting from January 1, 2010.

"""
df = pd.read_csv('Week17/data.csv', parse_dates=['Date'], index_col='Date')
print(df.head())

"""
3. Preprocessing the Data

MinMaxScaler(): This scales the data to a range of 0 to 1. This is important because neural networks perform better when input features are scaled properly.

"""

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df.values)


""""
4. Preparing Data for GRU
create_dataset(): Prepares the dataset for time-series forecasting. It creates sliding windows of time_step length to predict the next time step.
X.reshape(): Reshapes the input data to fit the expected shape for the GRU which is 3D: [samples, time steps, features].
"""
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0]) 
        y.append(data[i + time_step, 0]) 
    return np.array(X), np.array(y)

time_step = 100 
X, y = create_dataset(scaled_data, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)


"""
5. Building the GRU Model
GRU(units=50): Adds a GRU layer with 50 units (neurons).
return_sequences=True: Ensures that the GRU layer returns the entire sequence (required for stacking multiple GRU layers).
Dense(units=1): The output layer which predicts a single value for the next time step.
Adam(): An adaptive optimizer commonly used in deep learning.

"""
model = Sequential()
model.add(GRU(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(GRU(units=50))
model.add(Dense(units=1)) 

METRICS = metrics=['accuracy', 
                   Precision(name='precision'),
                   Recall(name='recall')]

model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics = METRICS)

""""
model.fit(): Trains the model on the prepared dataset. The epochs=10 specifies the number of iterations over the entire dataset, and batch_size=32 defines the number of samples per batch.

"""
model.fit(X, y, epochs=10, batch_size=32)

""""7. Making Predictions
Input Sequence: The code takes the last 100 temperature values from the dataset (scaled_data[-time_step:]) as an input sequence.
Reshaping the Input Sequence: The input sequence is reshaped into the shape (1, time_step, 1) because the GRU model expects a 3D input: [samples, time_steps, features]. Here samples=1 because we are making one prediction, time_steps=100 (the length of the input sequence) and features=1 because we are predicting only the temperature value.
model.predict(): Uses the trained model to predict future values based on the input data.

"""

input_sequence = scaled_data[-time_step:].reshape(1, time_step, 1)
predicted_values = model.predict(input_sequence)


""""8. Inverse Transforming the Predictions
Inverse Transforming the Predictions refers to the process of converting the scaled (normalized) predictions back to their original scale."""

predicted_values = scaler.inverse_transform(predicted_values)
print(f"The predicted temperature for the next day is: {predicted_values[0][0]:.2f}°C")
