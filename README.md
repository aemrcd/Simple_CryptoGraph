# Crypto Buy Prediction App

This repository contains a Python application for analyzing and predicting cryptocurrency prices using historical data. The application is built using Streamlit for the user interface, `yfinance` for fetching historical data, and `plotly` for interactive visualizations. It implements a Random Forest regression model for price prediction and provides technical analysis using indicators like Simple Moving Averages (SMA), Relative Strength Index (RSI), and Bollinger Bands.

## Features

- **Interactive User Interface**: Built with Streamlit, allowing users to select cryptocurrencies and visualize data.
- **Cryptocurrency Data Fetching**: Fetches historical data and current prices using `yfinance` and Binance API.
- **Technical Analysis**: Calculates and visualizes key technical indicators, including:
  - Simple Moving Averages (SMA)
  - Relative Strength Index (RSI)
  - Bollinger Bands
- **Price Prediction**: Utilizes a Random Forest regression model to predict the next day’s closing price.
- **Buy/Sell Recommendation**: Provides simple buy/sell recommendations based on SMA crossover strategies.

## Installation

### Prerequisites

Ensure you have Python installed (version 3.6 or later). You can check your Python version with the following command:

```bash
python --version
