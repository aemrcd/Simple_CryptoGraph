import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pycoingecko import CoinGeckoAPI

# Initialize the CoinGecko API
cg = CoinGeckoAPI()

# Set up the Streamlit layout
st.title("Cryptocurrency Price Comparison")
st.write("Select cryptocurrencies to compare their historical prices.")

# Get the list of popular coins from CoinGecko
coin_list = cg.get_coins_list()
coins = ["bitcoin", "ethereum", "litecoin", "ripple", "dogecoin"]

# Sidebar for user input
selected_coins = st.multiselect("Choose Cryptocurrencies", coins, default=["bitcoin", "ethereum"])

# Function to fetch historical data for a coin
def fetch_coin_data(coin_id):
    data = cg.get_coin_market_chart_by_id(id=coin_id, vs_currency="usd", days=30)
    prices = pd.DataFrame(data['prices'], columns=["timestamp", coin_id])
    prices["timestamp"] = pd.to_datetime(prices["timestamp"], unit='ms')
    return prices

# Fetch data for selected coins
if selected_coins:
    historical_data = pd.DataFrame()

    for coin in selected_coins:
        coin_data = fetch_coin_data(coin)
        if historical_data.empty:
            historical_data = coin_data
        else:
            historical_data = pd.merge(historical_data, coin_data, on="timestamp")

    # Plotting the graph
    plt.figure(figsize=(10, 5))
    for coin in selected_coins:
        plt.plot(historical_data["timestamp"], historical_data[coin], label=coin.capitalize())

    plt.title("Cryptocurrency Prices (Last 30 Days)")
    plt.xlabel("Date")
    plt.ylabel("Price in USD")
    plt.legend()

    # Display the graph in Streamlit
    st.pyplot(plt)

# Optional: show the raw data in the app
if st.checkbox("Show raw data"):
    st.write(historical_data)
