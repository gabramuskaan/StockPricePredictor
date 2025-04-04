import yfinance as yf
import pandas as pd
from datetime import datetime

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
            print(f"Successfully downloaded data for {ticker}")
        except Exception as e:
            print(f"Error downloading data for {ticker}: {e}")

    return data

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
