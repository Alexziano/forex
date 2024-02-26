import pandas as pd
import matplotlib.pyplot as plt
import requests

# Function to fetch historical price data for a symbol from Alpha Vantage
def fetch_historical_data(symbol, api_key, interval='30min', outputsize='compact'):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}&outputsize={outputsize}&apikey={api_key}"
    response = requests.get(url)
    data = response.json()
    return data

# Your Alpha Vantage API key
api_key = "6UC8C12GSPEVUHVD"

# Fetch historical data for EURUSD at 30-minute intervals
historical_data = fetch_historical_data("EURUSD", api_key, interval='30min')

# Extracting data from the API response
prices = historical_data['Time Series (30min)']
price_data = [{'Timestamp': timestamp, 'Price': float(data['4. close'])} for timestamp, data in prices.items()]

# Convert price data to a pandas DataFrame
df = pd.DataFrame(price_data)
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.set_index('Timestamp', inplace=True)

# Calculate rolling range (e.g., 10-period range)
rolling_high = df['Price'].rolling(window=10).max()
rolling_low = df['Price'].rolling(window=10).min()

# Plot price data and rolling range
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Price'], label='Price')
plt.plot(df.index, rolling_high, label='Rolling High', linestyle='--')
plt.plot(df.index, rolling_low, label='Rolling Low', linestyle='--')
plt.title('Price and Rolling Range')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# Identify consolidation periods
consolidation_periods = df[(df['Price'] <= rolling_high) & (df['Price'] >= rolling_low)]
print("Consolidation periods:")
print(consolidation_periods)
