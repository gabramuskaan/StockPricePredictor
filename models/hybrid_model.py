import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

class HybridModel(BaseEstimator, RegressorMixin):
    """
    Hybrid model that combines predictions from multiple models
    """

    def __init__(self, models, weights=None, adaptive=False, window_size=10):
        """
        Initialize the hybrid model
        """
        self.models = models
        self.model_names = list(models.keys())

        if weights is None:
            # Equal weighting
            self.weights = {name: 1/len(models) for name in self.model_names}
        else:
            self.weights = weights

        self.adaptive = adaptive
        self.window_size = window_size
        self.prediction_errors = {name: [] for name in self.model_names}

    def fit(self, X, y):
        """
        Fit all base models
        """
        for name, model in self.models.items():
            print(f"Training {name} model...")
            model.fit(X, y)
        return self

    def predict(self, X):
        """
        Make predictions using the hybrid model
        """
        predictions = {}

        for name, model in self.models.items():
            predictions[name] = model.predict(X)

        if self.adaptive and all(len(errors) >= self.window_size for errors in self.prediction_errors.values()):
            self._update_weights()

        # Weighted average of predictions
        weighted_pred = np.zeros_like(predictions[self.model_names[0]])
        for name in self.model_names:
            weighted_pred += self.weights[name] * predictions[name]

        return weighted_pred

    def _update_weights(self):
        """
        Update weights based on recent prediction errors
        """
        recent_errors = {}

        for name in self.model_names:
            recent_errors[name] = np.mean(np.abs(self.prediction_errors[name][-self.window_size:]))

        # Inverse error weighting
        total_inverse_error = sum(1/err if err > 0 else float('inf') for err in recent_errors.values())

        for name in self.model_names:
            if recent_errors[name] > 0:
                self.weights[name] = (1/recent_errors[name]) / total_inverse_error
            else:
                # If error is 0, give this model full weight
                self.weights = {n: 0 for n in self.model_names}
                self.weights[name] = 1
                break
