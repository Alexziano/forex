import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import pandas_datareader as pdr
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Fetch stock data for Apple Inc. (AAPL) from Alpha Vantage
start_date = datetime.datetime(2010, 1, 1)
end_date = datetime.datetime(2024, 1, 1)

df = pdr.av.time_series.AVTimeSeriesReader('AAPL', api_key='6UC8C12GSPEVUHVD').read()
df = df.reset_index()

# Plot stock prices
plt.plot(df['close'])
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.title('Stock Price Over Time')
plt.show()
print(df.columns)

# Calculate moving averages
ma100 = df['close'].rolling(100).mean()
ma200 = df['close'].rolling(200).mean()
plt.figure(figsize=(12, 6))
plt.plot(df['close'])
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')


# Prepare data for training
data_training = pd.DataFrame(df['close'][0:int(len(df) * 0.70)])

# Normalize the data
scaler = MinMaxScaler()
data_training_array = scaler.fit_transform(data_training)

# Prepare input data for LSTM
x_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i - 100: i])
    y_train.append(data_training_array[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)






from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

model = Sequential()

model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=60, activation='relu', return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(units=80, activation='relu', return_sequences=True))
model.add(Dropout(0.4))

model.add(LSTM(units=120, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, epochs=50, batch_size=32)

# Now you can continue with testing or making predictions using the trained model.
# Keep in mind that this is a basic example, and further adjustments might be needed.

from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

model = Sequential()

model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=60, activation='relu', return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(units=80, activation='relu', return_sequences=True))
model.add(Dropout(0.4))

model.add(LSTM(units=120, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, epochs=20, batch_size=32)

# Now you can continue with testing or making predictions using the trained model.
# Keep in mind that this is a basic example, and further adjustments might be needed.


model.save('my_model.keras')


import pandas as pd
data_testing = pd.DataFrame(df['close'][int(len(df) * 0.70):])


data_testing.head()

print(type(past_100_days))
print(type(data_testing))

past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)



final_df.head()

from sklearn.preprocessing import MinMaxScaler

def scaler_fit_transform(data):
    # Assuming you want to use MinMaxScaler as an example
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data


 input_data = scaler_fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])


x_test, y_test = np.array(x_test), np.array(y_test)
print(x_test.shape)
print(y_test.shape)import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import pandas_datareader as pdr
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Fetch stock data for Apple Inc. (AAPL) from Alpha Vantage
start_date = datetime.datetime(2010, 1, 1)
end_date = datetime.datetime(2024, 1, 1)

df = pdr.av.time_series.AVTimeSeriesReader('AAPL', api_key='6UC8C12GSPEVUHVD').read()
df = df.reset_index()

# Plot stock prices
plt.plot(df['close'])
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.title('Stock Price Over Time')
plt.show()

# Calculate moving averages
ma100 = df['close'].rolling(100).mean()
ma200 = df['close'].rolling(200).mean()
plt.figure(figsize=(12, 6))
plt.plot(df['close'])
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.show()

# Prepare data for training
data_training = pd.DataFrame(df['close'][0:int(len(df) * 0.70)])

# Normalize the data
scaler = MinMaxScaler()
data_training_array = scaler.fit_transform(data_training)

# Prepare input data for LSTM
x_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i - 100: i])
    y_train.append(data_training_array[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)




from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

model = Sequential()

model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=60, activation='relu', return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(units=80, activation='relu', return_sequences=True))
model.add(Dropout(0.4))

model.add(LSTM(units=120, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, epochs=50, batch_size=32)

# Now you can continue with testing or making predictions using the trained model.
# Keep in mind that this is a basic example, and further adjustments might be needed.
