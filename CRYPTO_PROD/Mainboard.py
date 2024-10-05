import streamlit as st
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import joblib
import plotly.graph_objects as go

class CryptoAnalyzer:
    def __init__(self, ticker):
        self.ticker = ticker
        self.data = None
        self.model = None

    # Fetch historical crypto data from yfinance
    def get_crypto_data(self):
        data = yf.Ticker(self.ticker)
        self.data = data.history(period="1y")
        return self.data

    # Get current price from Binance
    @staticmethod
    def get_current_price(symbol):
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            return float(data['price'])
        else:
            st.error(f"Error fetching data from Binance API: {response.status_code}")
            return None

    # Display crypto information
    def display_info(self):
        crypto_data = yf.Ticker(self.ticker)
        info = crypto_data.info
        
        st.subheader(f"Crypto Info: {info.get('name', 'N/A')}")
        st.write(f"**Symbol**: {info.get('symbol', 'N/A')}")
        st.write(f"**Sector**: {info.get('sector', 'N/A')}")
        st.write(f"**Market Cap**: ${info.get('marketCap', 'N/A')}")
        
        # Fetch current price from Binance
        binance_symbol = self.ticker.replace('-', '').upper() + 'USDT'  # Convert ticker for Binance
        current_price = self.get_current_price(binance_symbol)
        if current_price:
            st.write(f"**Current Price**: ${current_price:.2f}")
        
        st.write(f"**52 Week High**: ${info.get('fiftyTwoWeekHigh', 'N/A')}")
        st.write(f"**52 Week Low**: ${info.get('fiftyTwoWeekLow', 'N/A')}")
        st.write(f"**Volume**: {info.get('volume', 'N/A')}")
        st.write(f"**Currency**: {info.get('currency', 'N/A')}")

    # Generate technical indicators
    def generate_indicators(self):
        self.data['SMA_20'] = self.data['Close'].rolling(window=20).mean()
        self.data['SMA_50'] = self.data['Close'].rolling(window=50).mean()
        self.data['RSI'] = self.calculate_rsi(self.data['Close'])
        self.data = self.generate_bollinger_bands(self.data)

    # RSI calculation helper
    @staticmethod
    def calculate_rsi(series, period=14):
        delta = series.diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    # Generate Bollinger Bands
    @staticmethod
    def generate_bollinger_bands(data):
        data['Bollinger_High'] = data['Close'].rolling(window=20).mean() + (data['Close'].rolling(window=20).std() * 2)
        data['Bollinger_Low'] = data['Close'].rolling(window=20).mean() - (data['Close'].rolling(window=20).std() * 2)
        return data

    # Train the prediction model
    def train_model(self):
        # Using historical close prices to predict future ones
        self.data['Target'] = self.data['Close'].shift(-1)
        self.data = self.data.dropna()
        
        X = self.data[['Close', 'SMA_20', 'SMA_50', 'RSI']]
        y = self.data['Target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        st.write(f"Model MSE: {mse}")
        
        # Save the trained model
        joblib.dump(self.model, f'{self.ticker}_model.pkl')

    # Load the trained model
    def load_model(self):
        self.model = joblib.load(f'{self.ticker}_model.pkl')

    # Plot crypto data with indicators using Plotly
    def plot_crypto_data(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.data.index, y=self.data['Close'], mode='lines', name='Close Price'))
        fig.add_trace(go.Scatter(x=self.data.index, y=self.data['SMA_20'], mode='lines', name='SMA 20'))
        fig.add_trace(go.Scatter(x=self.data.index, y=self.data['SMA_50'], mode='lines', name='SMA 50'))
        fig.add_trace(go.Scatter(x=self.data.index, y=self.data['Bollinger_High'], mode='lines', name='Bollinger High'))
        fig.add_trace(go.Scatter(x=self.data.index, y=self.data['Bollinger_Low'], mode='lines', name='Bollinger Low'))
        st.plotly_chart(fig)

    # Make predictions
    def predict_next_day(self):
        last_row = self.data.iloc[-1]
        features = np.array([[last_row['Close'], last_row['SMA_20'], last_row['SMA_50'], last_row['RSI']]])
        prediction = self.model.predict(features)
        st.write(f"Predicted next day closing price: ${prediction[0]:.2f}")
        return prediction

    # Provide buy recommendation based on SMA crossover
    def buy_recommendation(self):
        last_row = self.data.iloc[-1]
        if last_row['SMA_20'] > last_row['SMA_50']:
            st.write("Recommendation: **BUY** - Short-term trend looks bullish (SMA 20 > SMA 50).")
        else:
            st.write("Recommendation: **HOLD** or **SELL** - Short-term trend looks bearish (SMA 20 <= SMA 50).")

# Streamlit app interface
st.title("Crypto Buy Prediction App")

# Select cryptocurrency
crypto_symbols = ["BTC-USD", "ETH-USD", "DOGE-USD"]
crypto_symbol = st.selectbox("Select Cryptocurrency", crypto_symbols)

if crypto_symbol in crypto_symbols:
    analyzer = CryptoAnalyzer(crypto_symbol)

    with st.spinner("Fetching data..."):
        crypto_data = analyzer.get_crypto_data()
        analyzer.display_info()
        
    analyzer.generate_indicators()
    
    st.subheader(f"{crypto_symbol} Historical Data with Indicators")
    analyzer.plot_crypto_data()

    st.subheader("Prediction Model")
    analyzer.train_model()

    # Load model if exists
    try:
        analyzer.load_model()
    except FileNotFoundError:
        st.warning("Model not found, training a new model.")
    
    # Make predictions
    st.subheader("Next Day Prediction")
    prediction = analyzer.predict_next_day()

    # Buy Recommendation
    st.subheader("Buy Recommendation")
    analyzer.buy_recommendation()
else:
    st.error("Invalid cryptocurrency selected.")
