# Stock Price Predictor

## Overview

This project is a Stock Price Prediction system built using multiple machine learning models including GRU, LSTM, KNN, Random Forest, SVR, and hybrid models. It leverages historical stock data to forecast future prices. The project includes data loading, preprocessing, model training, and a Streamlit-based web app for interactive predictions and visualization.

## Features

- Multiple base models: GRU, LSTM, KNN, Random Forest, SVR
- Hybrid and enhanced hybrid models combining base models for improved accuracy
- Data preprocessing and feature engineering for stock price data
- Streamlit web app for easy interaction and visualization of predictions
- Modular code structure for easy extension and experimentation

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd StockPricePredictor
   ```

2. Create and activate a Python virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage
### Running the Streamlit App
Launch the Streamlit app to interact with the models and visualize predictions:
```
streamlit run streamlit_app.py
```

## Project Structure
- `app/` - Contains the main app script (`app.py`) for running the application logic.
- `models/` - Contains model implementations:
  - `base_models/` - Individual base models (GRU, LSTM, KNN, Random Forest, SVR)
  - `hybrid_model.py` - Hybrid model combining base models
  - `enhanced_hybrid_model.py` - Enhanced hybrid model for improved performance
  - `data_preprocessing.py` - Data preprocessing utilities
  - `model_utils.py` - Helper functions for model training and evaluation
- `src/data/` - Data loading and processing scripts
  - `data_loader.py` - Loading raw stock data
  - `data_processor.py` - Processing and feature engineering
- `.streamlit/config.toml` - Streamlit configuration file
- `streamlit_app.py` - Entry point for the Streamlit web app
- `requirements.txt` - Python dependencies
