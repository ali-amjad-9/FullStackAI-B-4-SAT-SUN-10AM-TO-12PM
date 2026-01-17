# NAME: Muhammad Ali Amjad 
# SECTION B: QUESTION 2  
# STOCK PRICE FORECASTING Using DEEP LEARNING


# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing sklearn 
from sklearn.preprocessing import MinMaxScaler

# importing keras models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense, Dropout

#
from tensorflow.keras.callbacks import EarlyStopping


# STEP 1: LOADING THE DATASET
print(" STEP 1:LOADING DATASET: ")

# here we will use the csv file

filename = 'SectionB-Q2-Adobe_Data.csv'
df = pd.read_csv(filename)

#data tis loaded
print(df.head())

# conterting Date column to datetime
print("\nConverting...")
df['Date'] = pd.to_datetime(df['Date'])

# ordering up-to-date
df = df.sort_values('Date')
print("Data ordered")

# Extracting the values as a numpy array
data_values = df[['Open', 'High', 'Low', 'Close', 'Volume']].values
print(f"Total rows of data: {len(data_values)}")


# STEP 2: SPLIT DATA 


print("\n STEP 2:splitting DATA ")

# We can't use train_test_split from sklearn here 

split_index = int(len(data_values) * 0.8)

train_data = data_values[:split_index]
test_data = data_values[split_index:]

print(f"Training Data Rows: {len(train_data)}")
print(f"Testing Data Rows: {len(test_data)}")

# STEP 3: SCALING THE DATA


print("\n STEP 3: SCALING THE DATA")


# i think 0,1 is better for neural networks

scaler = MinMaxScaler(feature_range=(0, 1))

#
train_scaled = scaler.fit_transform(train_data)

# lets transform test data
test_scaled = scaler.transform(test_data)
print("Data scaled successfully.")


# STEP 4: 
print("\n STEP 4...")

#look backing 60 days back
lookback = 60

# here we will prepare training data
X_train = []
y_train = []

# Loop
for i in range(lookback, len(train_scaled)):
   
    X_train.append(train_scaled[i-lookback:i])
    
    
    y_train.append(train_scaled[i, 3])

# conversation of list to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)

# test data
X_test = []
y_test = []

for i in range(lookback, len(test_scaled)):
    X_test.append(test_scaled[i-lookback:i])
    y_test.append(test_scaled[i, 3])

X_test = np.array(X_test)
y_test = np.array(y_test)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")


# STEP 5:
print("\n STEP 5... ")

#RNN MODEL
model_rnn = Sequential()


# Layer 1: SimpleRNN
model_rnn.add(SimpleRNN(units=50, return_sequences=True, input_shape=(X_train.shape[1], 5)))
model_rnn.add(Dropout(0.2)) 

# Layer 2: SimpleRNN
model_rnn.add(SimpleRNN(units=50, return_sequences=False))
model_rnn.add(Dropout(0.2))

# quantity of Layer
model_rnn.add(Dense(units=1))

# compliling the model
model_rnn.compile(optimizer='adam', loss='mean_squared_error')

# Training the model

stopper = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

print("Starting training for RNN ...")
model_rnn.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test), callbacks=[stopper])

# RNN
predictions_rnn = model_rnn.predict(X_test)


#STEP 6: model 2
model_lstm = Sequential()
# Layer 1: LSTM
model_lstm.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 5)))
model_lstm.add(Dropout(0.2))

# Layer 2: LSTM
model_lstm.add(LSTM(units=50, return_sequences=False))
model_lstm.add(Dropout(0.2))

# Output Layer
model_lstm.add(Dense(units=1))

# Compiling
model_lstm.compile(optimizer='adam', loss='mean_squared_error')

print("Starting training for LSTM...")
model_lstm.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test), callbacks=[stopper])

# prediction
predictions_lstm = model_lstm.predict(X_test)



# STEP 7: converting back to original prices


print("\n STEP 7:converting back to original prices")

# we need to convert the scaled predictions back to original prices




#quantity

def inverse_transform_prices(preds):
    dummy_table = np.zeros((len(preds), 5))
    dummy_table[:, 3] = preds.flatten()
    real_values = scaler.inverse_transform(dummy_table)[:, 3]
    return real_values



# Converting everything
real_actual_prices = inverse_transform_prices(y_test)
real_rnn_prices = inverse_transform_prices(predictions_rnn)
real_lstm_prices = inverse_transform_prices(predictions_lstm)

print("Prices converted back to USD.")
print(f"Example Actual Price: {real_actual_prices[0]}")
print(f"Example LSTM Prediction: {real_lstm_prices[0]}")



# STEP 8


print("\n STEP 8...")


# CALCULATION
def calculate_accuracy(actual, predicted):
    correct_guesses = 0
    total_guesses = 0
    
    # loop starting from 1 
    for i in range(1, len(actual)):
        actual_move = actual[i] - actual[i-1]
        pred_move = predicted[i] - actual[i-1]
        
        # Checking if the signs match 
        if (actual_move > 0 and pred_move > 0):
            correct_guesses += 1
        elif (actual_move < 0 and pred_move < 0):
            correct_guesses += 1
            
        total_guesses += 1
        
    accuracy = (correct_guesses / total_guesses) * 100
    return accuracy

rnn_acc = calculate_accuracy(real_actual_prices, real_rnn_prices)
lstm_acc = calculate_accuracy(real_actual_prices, real_lstm_prices)

print(f"Simple RNN Directional Accuracy: {rnn_acc:.2f}%")
print(f"LSTM Directional Accuracy:       {lstm_acc:.2f}%")


# STEP 9: FINAL PLOT

print("\n STEP 9: FINAL PLOT ")

plt.figure(figsize=(12, 6))


plt.plot(real_actual_prices, color='blue', label='Actual Adobe Price', linewidth=1.5)

# Plotting RNN 
plt.plot(real_rnn_prices, color='orange', label='Simple RNN', linestyle='--', alpha=0.7)

# Plotting LSTM 
plt.plot(real_lstm_prices, color='red', label='LSTM Prediction', linewidth=1.5)

plt.title('Stock Price Prediction: RNN vs LSTM')
plt.xlabel('Time (Days in Test Set)')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True)

plt.show()

print("\n End of Section B Question 2 ")