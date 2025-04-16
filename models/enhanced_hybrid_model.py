import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

class EnhancedHybridModel(BaseEstimator, RegressorMixin):
    """
    Enhanced hybrid model that selects and combines the best-performing models
    """

    def __init__(self, models, top_k=2, selection_metric='rmse', 
                 ensemble_method='weighted', validation_split=0.2,
                 adaptive_weights=True, window_size=10):
        """
        Initialize the enhanced hybrid model

        Parameters:
        -----------
        models : dict
            Dictionary of models {name: model_instance}
        top_k : int, default=2
            Number of top models to include in the ensemble
        selection_metric : str, default='rmse'
            Metric to use for model selection ('rmse', 'mae', 'r2')
        ensemble_method : str, default='weighted'
            Method to combine models ('simple', 'weighted', 'stacked')
        validation_split : float, default=0.2
            Fraction of training data to use for validation
        adaptive_weights : bool, default=True
            Whether to adaptively adjust weights based on recent performance
        window_size : int, default=10
            Number of recent predictions to consider for adaptive weighting
        """
        self.models = models
        self.model_names = list(models.keys())
        self.top_k = min(top_k, len(models))
        self.selection_metric = selection_metric
        self.ensemble_method = ensemble_method
        self.validation_split = validation_split
        self.adaptive_weights = adaptive_weights
        self.window_size = window_size
        
        # Initialize model performance tracking
        self.model_performance = {}
        self.selected_models = []
        self.weights = {}
        
        # For adaptive weighting
        self.prediction_errors = {name: [] for name in self.model_names}
        self.recent_y_true = []
        self.recent_predictions = {name: [] for name in self.model_names}

    def fit(self, X, y):
        """
        Fit all base models and select the best ones
        """
        # Split data for validation
        val_size = int(len(X) * self.validation_split)
        if val_size > 0:
            X_train, X_val = X[:-val_size], X[-val_size:]
            y_train, y_val = y[:-val_size], y[-val_size:]
        else:
            X_train, X_val = X, X
            y_train, y_val = y, y
        
        # Train all models
        print("Training base models...")
        for name, model in self.models.items():
            print(f"Training {name} model...")
            model.fit(X_train, y_train)
            
            # Evaluate on validation set
            val_pred = model.predict(X_val)
            
            # Calculate metrics
            mse = mean_squared_error(y_val, val_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_val, val_pred)
            r2 = r2_score(y_val, val_pred)
            
            self.model_performance[name] = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }
            
            print(f"{name} validation RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
        # Select top models
        self._select_top_models()
        
        # Initialize weights
        self._initialize_weights()
        
        return self
    
    def _select_top_models(self):
        """
        Select the top-k models based on the chosen metric
        """
        if self.selection_metric == 'rmse':
            sorted_models = sorted(self.model_performance.items(), 
                                  key=lambda x: x[1]['rmse'])
        elif self.selection_metric == 'mae':
            sorted_models = sorted(self.model_performance.items(), 
                                  key=lambda x: x[1]['mae'])
        elif self.selection_metric == 'r2':
            # For R², higher is better, so we reverse the sort
            sorted_models = sorted(self.model_performance.items(), 
                                  key=lambda x: -x[1]['r2'])
        else:
            raise ValueError(f"Unknown selection metric: {self.selection_metric}")
        
        # Get top k models
        self.selected_models = [name for name, _ in sorted_models[:self.top_k]]
        
        print(f"Selected top {self.top_k} models: {self.selected_models}")
    
    def _initialize_weights(self):
        """
        Initialize weights for the selected models
        """
        if self.ensemble_method == 'simple':
            # Equal weights
            self.weights = {name: 1/len(self.selected_models) for name in self.selected_models}
        
        elif self.ensemble_method == 'weighted':
            # Inverse error weighting
            if self.selection_metric == 'rmse':
                metric_values = {name: self.model_performance[name]['rmse'] 
                               for name in self.selected_models}
            elif self.selection_metric == 'mae':
                metric_values = {name: self.model_performance[name]['mae'] 
                               for name in self.selected_models}
            elif self.selection_metric == 'r2':
                # For R², we use 1-R² as the "error"
                metric_values = {name: 1 - self.model_performance[name]['r2'] 
                               for name in self.selected_models}
            
            # Calculate inverse error
            inverse_errors = {name: 1/err if err > 0 else float('inf') 
                             for name, err in metric_values.items()}
            
            # Normalize to get weights
            total = sum(inverse_errors.values())
            self.weights = {name: val/total for name, val in inverse_errors.items()}
        
        else:  # 'stacked' or other methods
            # Default to equal weights
            self.weights = {name: 1/len(self.selected_models) for name in self.selected_models}
        
        print("Model weights initialized:")
        for name, weight in self.weights.items():
            print(f"{name}: {weight:.4f}")
    
    def predict(self, X):
        """
        Make predictions using the ensemble of selected models
        """
        if not self.selected_models:
            raise ValueError("No models selected. Call fit() first.")
        
        predictions = {}
        
        # Get predictions from each selected model
        for name in self.selected_models:
            predictions[name] = self.models[name].predict(X)
            
            # Store predictions for later evaluation (only for single predictions)
            if len(X) == 1 and self.adaptive_weights:
                self.recent_predictions[name].append(predictions[name][0])
                
                # Keep only the most recent predictions
                if len(self.recent_predictions[name]) > self.window_size:
                    self.recent_predictions[name].pop(0)
        
        # Update weights if adaptive and we have enough data
        if self.adaptive_weights and len(self.recent_y_true) >= self.window_size:
            self._update_weights()
        
        # Combine predictions
        if self.ensemble_method in ['simple', 'weighted']:
            # Weighted average
            weighted_pred = np.zeros_like(predictions[self.selected_models[0]])
            for name in self.selected_models:
                weighted_pred += self.weights[name] * predictions[name]
            
            return weighted_pred
        
        else:  # Default to simple average
            return np.mean([predictions[name] for name in self.selected_models], axis=0)
    
    def update_true_values(self, y_true):
        """
        Update with actual values to calculate errors for adaptive weighting
        """
        if not self.adaptive_weights:
            return
            
        if isinstance(y_true, (list, np.ndarray)):
            self.recent_y_true.extend(y_true)
        else:
            self.recent_y_true.append(y_true)
        
        # Keep only the most recent values
        if len(self.recent_y_true) > self.window_size:
            self.recent_y_true = self.recent_y_true[-self.window_size:]
        
        # Calculate errors for each model
        if len(self.recent_y_true) > 0:
            for name in self.selected_models:
                if len(self.recent_predictions[name]) == len(self.recent_y_true):
                    error = mean_squared_error(
                        self.recent_y_true,
                        self.recent_predictions[name]
                    )
                    self.prediction_errors[name].append(error)
                    
                    # Keep only the most recent errors
                    if len(self.prediction_errors[name]) > self.window_size:
                        self.prediction_errors[name].pop(0)
    
    def _update_weights(self):
        """
        Update weights based on recent prediction errors
        """
        recent_errors = {}
        
        for name in self.selected_models:
            if self.prediction_errors[name]:
                recent_errors[name] = np.mean(self.prediction_errors[name][-self.window_size:])
            else:
                recent_errors[name] = float('inf')  # High error if no data
        
        # Inverse error weighting
        total_inverse_error = sum(1/err if err > 0 else float('inf') 
                                 for err in recent_errors.values())
        
        if total_inverse_error > 0:
            for name in self.selected_models:
                if recent_errors[name] > 0:
                    self.weights[name] = (1/recent_errors[name]) / total_inverse_error
                else:
                    # If error is 0, give this model full weight
                    self.weights = {n: 0 for n in self.selected_models}
                    self.weights[name] = 1
                    break
    
    def get_model_weights(self):
        """
        Return the current weights of each model
        """
        return self.weights
    
    def get_model_performance(self):
        """
        Return the performance metrics of all models
        """
        return self.model_performance
    
    def plot_model_comparison(self):
        """
        Plot comparison of model performance
        """
        if not self.model_performance:
            raise ValueError("No model performance data. Call fit() first.")
        
        metrics = ['rmse', 'mae', 'r2']
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, metric in enumerate(metrics):
            values = [self.model_performance[name][metric] for name in self.model_names]
            axes[i].bar(self.model_names, values)
            axes[i].set_title(f'Model Comparison - {metric.upper()}')
            axes[i].set_xlabel('Model')
            axes[i].set_ylabel(metric.upper())
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
