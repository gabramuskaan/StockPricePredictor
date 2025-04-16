# models/model_utils.py
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(y_true, y_pred):
    """
    Evaluate model performance using multiple metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape
    }

def calculate_volatility(data, window=20):
    """
    Calculate volatility index based on stock data
    """
    returns = data['Close'].pct_change()
    volatility = returns.rolling(window=window).std() * np.sqrt(window)
    return volatility.iloc[-1] if not volatility.empty else 0

def select_best_model(models, X_val, y_val):
    """
    Select the best performing model based on validation data
    """
    best_model = None
    best_score = float('inf')

    for name, model in models.items():
        predictions = model.predict(X_val)
        mse = mean_squared_error(y_val, predictions)

        if mse < best_score:
            best_score = mse
            best_model = name

    return best_model
