import numpy as np
from models.base_models.svr import SVRModel

def test_svr_train_and_predict():
    X_train = np.array([[1], [2], [3]])
    y_train = np.array([1, 2, 3])
    X_test = np.array([[4]])
    model = SVRModel()
    # Use fit() instead of train()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    assert prediction.shape == (1,)
