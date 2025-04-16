# models/base_models/random_forest.py
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, RegressorMixin

class RandomForestModel(BaseEstimator, RegressorMixin):
    """
    Random Forest model for time series prediction
    """

    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None
        self.is_fitted = False

    def fit(self, X, y):
        """
        Fit the Random Forest model
        """
        # Reshape X if it's 3D (from sequence data)
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], -1)  # Flatten the sequence dimension

        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1  # Use all available cores
        )
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X):
        """
        Make predictions using the Random Forest model
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")

        # Reshape X if it's 3D (from sequence data)
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], -1)  # Flatten the sequence dimension

        return self.model.predict(X)

    def feature_importance(self):
        """
        Get feature importance from the model
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")

        return self.model.feature_importances_
