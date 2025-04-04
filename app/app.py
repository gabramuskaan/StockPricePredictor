gabraimport streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import custom modules
from src.data.data_loader import load_stock_data, get_available_tickers
from src.data.data_processor import preprocess_data, train_test_split
from models.base_models.lstm import LSTMModel
from models.hybrid_model import HybridModel

def main():
    st.set_page_config(
        page_title="Stock Market Predictor",
        page_icon="📈",
        layout="wide"
    )

    # Sidebar navigation
    st.sidebar.title("Stock Market Predictor")
    page = st.sidebar.selectbox(
        "Navigation",
        ["Home", "Company Analysis", "Prediction", "Comparison"]
    )

    # Page routing
    if page == "Home":
        home_page()
    elif page == "Company Analysis":
        company_analysis_page()
    elif page == "Prediction":
        prediction_page()
    elif page == "Comparison":
        comparison_page()

def home_page():
    st.title("Stock Market Predictor")
    st.write("""
    ## Welcome to the Stock Market Predictor App

    This application uses machine learning to predict stock prices for major Indian companies.

    ### Features:
    - Analyze historical stock data
    - Compare multiple companies
    - Predict future stock prices using a hybrid model
    - Visualize trends and patterns

    Use the sidebar to navigate between different sections of the app.
    """)

    # Show recent market overview
    st.subheader("Recent Market Overview")
    tickers = get_available_tickers()[:5]  # Get top 5 tickers
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

    data = load_stock_data(tickers, start_date, end_date)

    # Display recent performance
    for ticker, df in data.items():
        if not df.empty:
            change = ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100
            color = "green" if change >= 0 else "red"
            st.markdown(f"**{ticker}**: {df['Close'].iloc[-1]:.2f} INR "
                      f"<span style='color:{color}'>{change:.2f}%</span>", unsafe_allow_html=True)

def company_analysis_page():
    st.title("Company Analysis")

    # Company selection
    ticker = st.selectbox("Select Company", get_available_tickers())

    # Date range selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime(2020, 1, 1))
    with col2:
        end_date = st.date_input("End Date", datetime.now())

    # Load data
    data = load_stock_data([ticker], start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

    if ticker in data and not data[ticker].empty:
        df = data[ticker]

        # Display basic info
        st.subheader(f"{ticker} Stock Data")
        st.write(df.tail())

        # Plot stock prices
        st.subheader("Stock Price History")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index, df['Close'], label='Close Price')
        ax.set_title(f"{ticker} Close Price History")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (INR)")
        ax.legend()
        st.pyplot(fig)

        # Technical indicators
        st.subheader("Technical Indicators")
        processed_df = preprocess_data(df)

        # Plot moving averages
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(processed_df.index, processed_df['Close'], label='Close Price')
        ax.plot(processed_df.index, processed_df['MA5'], label='5-day MA')
        ax.plot(processed_df.index, processed_df['MA20'], label='20-day MA')
        ax.plot(processed_df.index, processed_df['MA50'], label='50-day MA')
        ax.set_title(f"{ticker} Moving Averages")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (INR)")
        ax.legend()
        st.pyplot(fig)

        # Volume chart
        st.subheader("Trading Volume")
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(df.index, df['Volume'])
        ax.set_title(f"{ticker} Trading Volume")
        ax.set_xlabel("Date")
        ax.set_ylabel("Volume")
        st.pyplot(fig)
    else:
        st.error(f"No data available for {ticker}")

def prediction_page():
    st.title("Stock Price Prediction")

    # Company selection
    ticker = st.selectbox("Select Company", get_available_tickers())

    # Date range selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Training Data Start Date", datetime(2015, 1, 1))
    with col2:
        end_date = st.date_input("Training Data End Date", datetime.now())

    # Prediction days
    prediction_days = st.slider("Number of Days to Predict", 7, 90, 30)

    # Load data
    data = load_stock_data([ticker], start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

    if ticker in data and not data[ticker].empty:
        df = data[ticker]

        # Preprocess data
        processed_df = preprocess_data(df)
        processed_df = processed_df.dropna()

        if st.button("Generate Prediction"):
            with st.spinner("Training model and generating predictions..."):
                # Prepare data for model
                X_train, X_test, y_train, y_test, scaler = train_test_split(processed_df, train_size=0.8)

                # Train LSTM model
                model = LSTMModel(epochs=50)
                model.fit(X_train, y_train)

                # Make predictions
                test_predictions = model.predict(X_test)
                test_predictions = scaler.inverse_transform(test_predictions.reshape(-1, 1))

                # Generate future predictions
                last_sequence = X_test[-1].reshape(1, X_test.shape[1], 1)
                future_predictions = []

                for _ in range(prediction_days):
                    next_pred = model.predict(last_sequence)
                    future_predictions.append(next_pred[0])

                    # Update sequence for next prediction
                    last_sequence = np.append(last_sequence[:, 1:, :], [[next_pred]], axis=1)

                future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

                # Create future dates
                last_date = df.index[-1]
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=prediction_days)

                # Plot predictions
                st.subheader(f"Prediction for next {prediction_days} days")
                fig, ax = plt.subplots(figsize=(12, 6))

                # Plot historical data
                ax.plot(df.index[-100:], df['Close'][-100:], label='Historical Close Price')

                # Plot future predictions
                ax.plot(future_dates, future_predictions, label='Predicted Close Price', color='red')

                ax.set_title(f"{ticker} Stock Price Prediction")
                ax.set_xlabel("Date")
                ax.set_ylabel("Price (INR)")
                ax.legend()
                st.pyplot(fig)

                # Display prediction data
                prediction_df = pd.DataFrame({
                    'Date': future_dates,
                    'Predicted Close Price': future_predictions.flatten()
                })
                prediction_df.set_index('Date', inplace=True)
                st.write(prediction_df)
    else:
        st.error(f"No data available for {ticker}")

def comparison_page():
    st.title("Company Comparison")

    # Company selection
    tickers = st.multiselect("Select Companies to Compare", get_available_tickers(), default=['RELIANCE.NS', 'TCS.NS'])

    if not tickers:
        st.warning("Please select at least one company")
        return

    # Date range selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime(2020, 1, 1))
    with col2:
        end_date = st.date_input("End Date", datetime.now())

    # Load data
    data = load_stock_data(tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

    # Normalize data for comparison
    if st.checkbox("Normalize Data (for better comparison)"):
        for ticker in tickers:
            if ticker in data and not data[ticker].empty:
                data[ticker]['Close'] = data[ticker]['Close'] / data[ticker]['Close'].iloc[0] * 100
        ylabel = "Normalized Price (First day = 100)"
    else:
        ylabel = "Price (INR)"

    # Plot comparison
    st.subheader("Stock Price Comparison")
    fig, ax = plt.subplots(figsize=(12, 6))

    for ticker in tickers:
        if ticker in data and not data[ticker].empty:
            ax.plot(data[ticker].index, data[ticker]['Close'], label=ticker)

    ax.set_title("Stock Price Comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.legend()
    st.pyplot(fig)

    # Performance metrics
    st.subheader("Performance Metrics")
    metrics = []

    for ticker in tickers:
        if ticker in data and not data[ticker].empty:
            df = data[ticker]
            start_price = df['Close'].iloc[0]
            end_price = df['Close'].iloc[-1]
            change_pct = ((end_price - start_price) / start_price) * 100

            # Calculate volatility
            daily_returns = df['Close'].pct_change().dropna()
            volatility = daily_returns.std() * (252 ** 0.5) * 100  # Annualized volatility

            metrics.append({
                'Ticker': ticker,
                'Start Price': start_price,
                'End Price': end_price,
                'Change (%)': change_pct,
                'Volatility (%)': volatility
            })

    metrics_df = pd.DataFrame(metrics)
    st.write(metrics_df)

if __name__ == "__main__":
    main()
