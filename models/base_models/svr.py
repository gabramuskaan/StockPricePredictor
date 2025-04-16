# models/base_models/svr.py
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin

class SVRModel(BaseEstimator, RegressorMixin):
    """
    Support Vector Regression model for time series prediction
    """

    def __init__(self, kernel='rbf', C=1.0, epsilon=0.1, gamma='scale'):
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X, y):
        """
        Fit the SVR model
        """
        # Reshape X if it's 3D (from sequence data)
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], -1)  # Flatten the sequence dimension

        # Scale features for better SVR performance
        X_scaled = self.scaler.fit_transform(X)

        self.model = SVR(
            kernel=self.kernel,
            C=self.C,
            epsilon=self.epsilon,
            gamma=self.gamma
        )
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self

    def predict(self, X):
        """
        Make predictions using the SVR model
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")

        # Reshape X if it's 3D (from sequence data)
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], -1)  # Flatten the sequence dimension

        # Scale features
        X_scaled = self.scaler.transform(X)

        return self.model.predict(X_scaled)
