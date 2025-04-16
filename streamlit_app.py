# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import yfinance as yf

def get_available_tickers():
    """
    Returns a list of available stock tickers for the app
    """
    return [
        'RELIANCE.NS',  # Reliance Industries
        'TCS.NS',       # Tata Consultancy Services
        'HDFCBANK.NS',  # HDFC Bank
        'INFY.NS',      # Infosys
        'ICICIBANK.NS', # ICICI Bank
        'HINDUNILVR.NS',# Hindustan Unilever
        'SBIN.NS',      # State Bank of India
        'BAJFINANCE.NS',# Bajaj Finance
        'BHARTIARTL.NS',# Bharti Airtel
        'KOTAKBANK.NS'  # Kotak Mahindra Bank
    ]

def load_stock_data(tickers, start_date='2015-01-01', end_date=None):
    """
    Load stock data for multiple tickers
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    data = {}
    for ticker in tickers:
        try:
            data[ticker] = yf.download(ticker, start=start_date, end=end_date)
        except Exception as e:
            st.error(f"Error downloading data for {ticker}: {e}")

    return data

def main():
    st.set_page_config(
        page_title="Stock Market Predictor - CLEAN VERSION",
        page_icon="📈",
        layout="wide"
    )

    st.title("Stock Market Predictor - CLEAN VERSION")
    st.write("""
    ## Welcome to the Stock Market Predictor App

    This application uses machine learning to predict stock prices for major Indian companies.

    ### Features:
    - Analyze historical stock data
    - Compare multiple companies
    - Predict future stock prices using a hybrid model
    - Visualize trends and patterns
    """)

    # Show recent market overview
    st.subheader("Recent Market Overview")

    try:
        tickers = get_available_tickers()[:5]  # Get top 5 tickers
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

        data = load_stock_data(tickers, start_date, end_date)

        # Display recent performance
        for ticker, df in data.items():
            if not df.empty:
                # Calculate change safely
                first_price = float(df['Close'].iloc[0])
                last_price = float(df['Close'].iloc[-1])
                change = ((last_price - first_price) / first_price) * 100

                # Set color based on change
                color = "green" if change >= 0 else "red"

                # Display with formatting
                st.markdown(f"**{ticker}**: {last_price:.2f} INR "
                          f"<span style='color:{color}'>{change:.2f}%</span>", unsafe_allow_html=True)
            else:
                st.info(f"No recent data available for {ticker}")
    except Exception as e:
        st.error(f"Error loading market overview: {str(e)}")

if __name__ == "__main__":
    main()
