# models/base_models/lstm.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.base import BaseEstimator, RegressorMixin

class LSTMModel(BaseEstimator, RegressorMixin):
    """
    LSTM model for time series prediction
    """

    def __init__(self, seq_length=30, units=50, dropout=0.2, epochs=100, batch_size=32):
        self.seq_length = seq_length
        self.units = units
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None

    def build_model(self, input_shape):
        """
        Build the LSTM model architecture
        """
        model = Sequential()
        model.add(LSTM(units=self.units, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(self.dropout))
        model.add(LSTM(units=self.units, return_sequences=True))
        model.add(Dropout(self.dropout))
        model.add(LSTM(units=self.units))
        model.add(Dropout(self.dropout))
        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def fit(self, X, y):
        """
        Fit the LSTM model
        """
        # Reshape input to be [samples, time steps, features]
        if len(X.shape) < 3:
            X = X.reshape(X.shape[0], X.shape[1], 1)

        self.model = self.build_model((X.shape[1], X.shape[2]))
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=1)
        return self

    def predict(self, X):
        """
        Make predictions using the LSTM model
        """
        # Reshape input if needed
        if len(X.shape) < 3:
            X = X.reshape(X.shape[0], X.shape[1], 1)

        return self.model.predict(X).flatten()
