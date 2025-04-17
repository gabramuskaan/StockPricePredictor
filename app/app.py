import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

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

def evaluate_predictions(y_true, y_pred):
    """
    Evaluate predictions using multiple metrics
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import numpy as np
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Calculate MAPE (Mean Absolute Percentage Error)
    # Avoid division by zero
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape
    }

def home_page():
    st.title("Stock Market Predictor - UPDATED VERSION")  # Changed title to be obviously different
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

        # Only plot moving averages if they exist and have valid data
        if 'MA5' in processed_df.columns and not processed_df['MA5'].isna().all():
            ax.plot(processed_df.index, processed_df['MA5'], label='5-day MA')
        if 'MA20' in processed_df.columns and not processed_df['MA20'].isna().all():
            ax.plot(processed_df.index, processed_df['MA20'], label='20-day MA')
        if 'MA50' in processed_df.columns and not processed_df['MA50'].isna().all():
            ax.plot(processed_df.index, processed_df['MA50'], label='50-day MA')

        ax.set_title(f"{ticker} Moving Averages")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (INR)")
        ax.legend()
        st.pyplot(fig)

        # Volume chart - Fixed version
        st.subheader("Trading Volume")
        try:
            # Use pandas built-in plotting which handles DatetimeIndex correctly
            fig, ax = plt.subplots(figsize=(12, 4))
            df['Volume'].plot(kind='bar', ax=ax)
            ax.set_title(f"{ticker} Trading Volume")
            ax.set_xlabel("Date")
            ax.set_ylabel("Volume")
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            # Alternative approach if the first one fails
            try:
                fig, ax = plt.subplots(figsize=(12, 4))
                # Convert dates to strings for the x-axis
                ax.bar(df.index.astype(str), df['Volume'])
                ax.set_title(f"{ticker} Trading Volume")
                ax.set_xlabel("Date")
                ax.set_ylabel("Volume")
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            except Exception as e2:
                st.error(f"Error creating volume chart: {str(e2)}")
                st.write("Volume data preview:")
                st.write(df['Volume'].head())
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

    # Model selection
    st.subheader("Model Selection")
    
    col1, col2 = st.columns(2)
    with col1:
        models_to_use = st.multiselect(
            "Select Models to Include",
            ["LSTM", "Random Forest", "SVR", "KNN", "GRU"],
            default=["LSTM", "Random Forest", "SVR", "GRU"]
        )
    
    with col2:
        top_k = st.slider("Number of Top Models to Use in Hybrid", 1, 5, 2, 
                         help="The hybrid model will select this many top-performing models")
    
    # Advanced options
    with st.expander("Advanced Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            selection_metric = st.selectbox(
                "Model Selection Metric",
                ["rmse", "mae", "r2"],
                help="Metric used to select the best models"
            )
            
            ensemble_method = st.selectbox(
                "Ensemble Method",
                ["weighted", "simple"],
                help="Method to combine model predictions"
            )
        
        with col2:
            validation_split = st.slider(
                "Validation Split",
                0.1, 0.3, 0.2,
                help="Fraction of training data to use for model selection"
            )
            
            adaptive_weights = st.checkbox(
                "Adaptive Weights",
                True,
                help="Dynamically adjust model weights based on recent performance"
            )

    # Load data
    data = load_stock_data([ticker], start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

    if ticker in data and not data[ticker].empty:
        df = data[ticker]

        # Preprocess data
        from models.data_preprocessing import preprocess_data, train_test_split
        processed_df = preprocess_data(df)
        processed_df = processed_df.dropna()

        if st.button("Generate Prediction"):
            with st.spinner("Training models and generating predictions..."):
                try:
                    # Prepare data for model
                    X_train, X_test, y_train, y_test, scaler = train_test_split(processed_df, train_size=0.8)

                    # Initialize models based on user selection
                    models = {}

                    if "LSTM" in models_to_use:
                        from models.base_models.lstm import LSTMModel
                        models["LSTM"] = LSTMModel(epochs=50)

                    if "Random Forest" in models_to_use:
                        from models.base_models.random_forest import RandomForestModel
                        models["Random Forest"] = RandomForestModel(n_estimators=100)

                    if "SVR" in models_to_use:
                        from models.base_models.svr import SVRModel
                        models["SVR"] = SVRModel()

                    if "KNN" in models_to_use:
                        from models.base_models.knn import KNNModel
                        models["KNN"] = KNNModel(n_neighbors=5)

                    if "GRU" in models_to_use:
                        from models.base_models.gru import GRUModel
                        models["GRU"] = GRUModel(epochs=50)

                    if not models:
                        st.error("Please select at least one model.")
                        return

                    # Create enhanced hybrid model
                    from models.enhanced_hybrid_model import EnhancedHybridModel
                    hybrid_model = EnhancedHybridModel(
                        models, 
                        top_k=min(top_k, len(models)),
                        selection_metric=selection_metric,
                        ensemble_method=ensemble_method,
                        validation_split=validation_split,
                        adaptive_weights=adaptive_weights
                    )

                    # Train hybrid model (which trains all base models)
                    hybrid_model.fit(X_train, y_train)

                    # Add model comparison visualization
                    st.subheader("Model Performance Comparison")
                    try:
                        fig = hybrid_model.plot_model_comparison()
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Error plotting model comparison: {str(e)}")

                    # Then continue with making predictions
                    # Make predictions on test data
                    test_predictions = hybrid_model.predict(X_test)

                    # Show model performance comparison
                    st.subheader("Model Performance Comparison")
                    fig = hybrid_model.plot_model_comparison()
                    st.pyplot(fig)

                    # Make predictions on test data
                    test_predictions = hybrid_model.predict(X_test)

                    # If test_predictions is 1D, reshape it for inverse transform
                    if len(test_predictions.shape) == 1:
                        test_predictions = test_predictions.reshape(-1, 1)

                    # Debug information
                    st.write(f"Test predictions shape: {test_predictions.shape}")
                    st.write(f"Scaler min_ shape: {scaler.min_.shape}")
                    st.write(f"Scaler scale_ shape: {scaler.scale_.shape}")

                    # Create a proper dummy array with the same number of features as used during scaling
                    n_features = scaler.min_.shape[0]
                    dummy = np.zeros((len(test_predictions), n_features))
                    dummy[:, 0] = test_predictions.flatten()  # Assuming first column is Close price

                    # Inverse transform
                    try:
                        test_predictions_actual = scaler.inverse_transform(dummy)[:, 0]
                    except ValueError as e:
                        st.error(f"Error during inverse transform: {e}")
                        st.write("Attempting alternative approach...")

                        # Alternative approach: create a new scaler just for the Close price
                        from sklearn.preprocessing import MinMaxScaler
                        close_scaler = MinMaxScaler()
                        close_scaler.fit(processed_df['Close'].values.reshape(-1, 1))

                        # Use this scaler for the inverse transform
                        test_predictions_actual = close_scaler.inverse_transform(test_predictions)[:, 0]

                    # Evaluate model performance on test data
                    try:
                        # Get the actual closing prices for the test period
                        y_test_actual = df['Close'].values[-len(test_predictions_actual):]

                        # Make sure the arrays are the same length
                        min_len = min(len(y_test_actual), len(test_predictions_actual))
                        y_test_actual = y_test_actual[-min_len:]
                        test_predictions_actual = test_predictions_actual[-min_len:]

                        # Calculate evaluation metrics
                        metrics = evaluate_predictions(y_test_actual, test_predictions_actual)

                        # Display metrics in the app
                        st.subheader("Model Evaluation on Test Data")

                        # Format the metrics for better readability
                        formatted_metrics = {
                            'MSE': f"{metrics['MSE']:.2f}",
                            'RMSE': f"{metrics['RMSE']:.2f}",
                            'MAE': f"{metrics['MAE']:.2f}",
                            'R²': f"{metrics['R2']:.4f}",
                            'MAPE': f"{metrics['MAPE']:.2f}%"
                    }

                        # Create a DataFrame for display
                        metrics_df = pd.DataFrame({
                        'Metric': list(formatted_metrics.keys()),
                        'Value': list(formatted_metrics.values())
                    })

                        # Display as a table
                        st.table(metrics_df)

                        # Add interpretation
                        if metrics['MAPE'] < 5:
                            st.success("The model shows excellent accuracy with less than 5% average percentage error.")
                        elif metrics['MAPE'] < 10:
                            st.info("The model shows good accuracy with less than 10% average percentage error.")
                        else:
                            st.warning("The model shows moderate accuracy. Consider adjusting parameters or adding more training data.")

                    except Exception as e:
                        st.error(f"Error calculating evaluation metrics: {str(e)}")

                    # If test_predictions is 1D, reshape it for inverse transform
                    if len(test_predictions.shape) == 1:
                        test_predictions = test_predictions.reshape(-1, 1)

                    # Prepare for inverse transform
                    # We need to create a dummy array with the same shape as the original data
                    # Create a proper dummy array with the same number of features as used during scaling
                    n_features = scaler.min_.shape[0]  # This will be 9 based on your error
                    dummy = np.zeros((len(test_predictions), n_features))
                    dummy[:, 0] = test_predictions.flatten()  # Assuming first column is Close price

                    # Inverse transform
                    test_predictions_actual = scaler.inverse_transform(dummy)[:, 0]

                    # Generate future predictions
                    last_sequence = X_test[-1].reshape(1, X_test.shape[1], X_test.shape[2])
                    future_predictions = []

                    for _ in range(prediction_days):
                        next_pred = hybrid_model.predict(last_sequence)
                        future_predictions.append(next_pred[0])

                        # Update sequence for next prediction
                        # Create a new sequence by shifting and adding the new prediction
                        new_seq = np.copy(last_sequence)
                        new_seq[0, :-1, :] = new_seq[0, 1:, :]
                        new_seq[0, -1, 0] = next_pred[0]  # Assuming prediction is for the first feature
                        last_sequence = new_seq

                    # Convert future predictions to actual values
                    future_predictions = np.array(future_predictions).reshape(-1, 1)

                    # Create a proper dummy array for future predictions
                    n_features = scaler.min_.shape[0]
                    future_dummy = np.zeros((len(future_predictions), n_features))
                    future_dummy[:, 0] = future_predictions.flatten()

                    # Inverse transform future predictions
                    try:
                        future_predictions_actual = scaler.inverse_transform(future_dummy)[:, 0]
                    except ValueError as e:

                        # Alternative approach: create a new scaler just for the Close price
                        from sklearn.preprocessing import MinMaxScaler
                        close_prices = df['Close'].values.reshape(-1, 1)
                        close_scaler = MinMaxScaler()
                        close_scaler.fit(close_prices)

                        # Use this scaler for the inverse transform
                        future_predictions_actual = close_scaler.inverse_transform(future_predictions)[:, 0]

                    # Create future dates
                    last_date = df.index[-1]
                    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=prediction_days)

                    # Plot predictions
                    st.subheader(f"Prediction for next {prediction_days} days")
                    fig, ax = plt.subplots(figsize=(12, 6))

                    # Plot historical data
                    ax.plot(df.index[-100:], df['Close'][-100:], label='Historical Close Price')

                    # Plot future predictions
                    ax.plot(future_dates, future_predictions_actual, label='Predicted Close Price', color='red')

                    ax.set_title(f"{ticker} Stock Price Prediction")
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Price (INR)")
                    ax.legend()
                    st.pyplot(fig)

                    # Display prediction data
                    prediction_df = pd.DataFrame({
                        'Date': future_dates,
                        'Predicted Close Price': future_predictions_actual
                    })
                    prediction_df.set_index('Date', inplace=True)
                    st.write(prediction_df)

                    # Display model weights
                    st.subheader("Selected Models and Weights")
                    weights = hybrid_model.get_model_weights()
                    weights_df = pd.DataFrame({
                        'Model': list(weights.keys()),
                        'Weight': list(weights.values())
                    })

                    # Create a bar chart of weights
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.bar(weights_df['Model'], weights_df['Weight'])
                    ax.set_title("Model Weights in Hybrid Prediction")
                    ax.set_ylabel("Weight")
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)

                except Exception as e:
                    st.error(f"Error generating prediction: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
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
