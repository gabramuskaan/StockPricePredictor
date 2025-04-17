# models/data_preprocessing.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df):
    """
    Preprocess the stock data for model training

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing stock data with at least 'Close' column

    Returns:
    --------
    pandas.DataFrame
        Preprocessed DataFrame with features
    """
    # Make a copy to avoid modifying the original data
    processed_df = df.copy()

    # Calculate technical indicators
    # 1. Moving Averages
    processed_df['MA5'] = processed_df['Close'].rolling(window=5).mean()
    processed_df['MA20'] = processed_df['Close'].rolling(window=20).mean()

    # 2. Relative Strength Index (RSI)
    delta = processed_df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    processed_df['RSI'] = 100 - (100 / (1 + rs))

    # 3. Moving Average Convergence Divergence (MACD)
    ema12 = processed_df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = processed_df['Close'].ewm(span=26, adjust=False).mean()
    processed_df['MACD'] = ema12 - ema26
    processed_df['Signal_Line'] = processed_df['MACD'].ewm(span=9, adjust=False).mean()

    # 4. Bollinger Bands
    processed_df['20d_std'] = processed_df['Close'].rolling(window=20).std()
    processed_df['Upper_Band'] = processed_df['MA20'] + (processed_df['20d_std'] * 2)
    processed_df['Lower_Band'] = processed_df['MA20'] - (processed_df['20d_std'] * 2)

    # 5. Price Rate of Change
    processed_df['ROC'] = processed_df['Close'].pct_change(periods=5) * 100

    # Drop NaN values
    processed_df = processed_df.dropna()

    return processed_df

def train_test_split(df, train_size=0.8, sequence_length=60):
    """
    Split the data into training and testing sets and prepare sequences
    """
    # Select features
    features = ['Close', 'MA5', 'MA20', 'RSI', 'MACD', 'Signal_Line',
                'Upper_Band', 'Lower_Band', 'ROC']

    # Make sure we only use features that exist in the dataframe
    available_features = [f for f in features if f in df.columns]
    data = df[available_features].values

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Create sequences
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i, 0])  # 0 is the index of 'Close' price

    X, y = np.array(X), np.array(y)

    # Split into train and test sets
    train_size = int(len(X) * train_size)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    return X_train, X_test, y_train, y_test, scaler
