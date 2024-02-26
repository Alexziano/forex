

import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import pandas_datareader as pdr

# Fetch stock data for Apple Inc. (AAPL) from Alpha Vantage
start_date = datetime.datetime(2024, 1, 15)
end_date = datetime.datetime(2024, 1, 19)

df = pdr.av.time_series.AVTimeSeriesReader('AAPL', api_key='6UC8C12GSPEVUHVD').read()
df = df.reset_index()

# Update the timestamp column name based on your DataFrame
# For example, if the column name is 'date', replace 'timestamp' with 'date'
df['day'] = df['timestamp'].dt.date
df['hour'] = df['timestamp'].dt.hour

# Group by day and hour
grouped_data = df.groupby(['day', 'hour'])

# Create subplots for each day
fig, axes = plt.subplots(len(grouped_data), 1, figsize=(15, 5 * len(grouped_data)), sharex=True)

# Plot each day's data in a separate subplot
for (day_hour, day_hour_data), ax in zip(grouped_data, axes):
    ax.plot(day_hour_data['timestamp'], day_hour_data['close'], label=f'{day_hour[0]} {day_hour[1]}:00')
    ax.set_title(f'Day {day_hour[0]}')
    ax.legend()

plt.tight_layout()
plt.show()

