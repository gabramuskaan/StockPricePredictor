import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(y_true, y_pred):
    """Evaluate model performance using multiple metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

def calculate_volatility(data, window=20):
    """Calculate volatility index based on stock data"""
    returns = data['Close'].pct_change()
    volatility = returns.rolling(window=window).std() * np.sqrt(window)
    return volatility.iloc[-1] if not volatility.empty else 0
