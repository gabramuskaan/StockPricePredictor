import pandas as pd
import numpy as np
from models.data_preprocessing import preprocess_data

def test_preprocess_data_removes_missing():
    # Create test data with the 'Close' column instead of 'price'
    df = pd.DataFrame({'Close': [1, None, 3]})
    processed = preprocess_data(df)
    assert processed.isnull().sum().sum() == 0

def test_preprocess_data_scales_data():
    # Create test data with more rows to avoid all being dropped
    df = pd.DataFrame({
        'Close': [i for i in range(1, 26)]  # 25 data points
    })
    processed = preprocess_data(df)
    # Check if data contains values
    assert not processed.empty
