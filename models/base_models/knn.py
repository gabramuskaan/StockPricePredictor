# models/base_models/knn.py
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin

class KNNModel(BaseEstimator, RegressorMixin):
    """
    K-Nearest Neighbors model for time series prediction
    """

    def __init__(self, n_neighbors=5, weights='uniform', algorithm='auto'):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X, y):
        """
        Fit the KNN model
        """
        # Reshape X if it's 3D (from sequence data)
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], -1)  # Flatten the sequence dimension

        # Scale features for better KNN performance
        X_scaled = self.scaler.fit_transform(X)

        self.model = KNeighborsRegressor(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            algorithm=self.algorithm,
            n_jobs=-1  # Use all available cores
        )
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self

    def predict(self, X):
        """
        Make predictions using the KNN model
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")

        # Reshape X if it's 3D (from sequence data)
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], -1)  # Flatten the sequence dimension

        # Scale features
        X_scaled = self.scaler.transform(X)

        return self.model.predict(X_scaled)
