import numpy as np
from models.base_models.gru import GRUModel

def test_gru_train_and_predict():
    X_train = np.random.rand(10, 3, 1)
    y_train = np.random.rand(10)
    X_test = np.random.rand(2, 3, 1)
    # Initialize with correct parameters
    model = GRUModel(seq_length=3, units=50)
    # Use fit() instead of train()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    assert prediction.shape == (2,)
