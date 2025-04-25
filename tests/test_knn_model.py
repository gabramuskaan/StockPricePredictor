import numpy as np
from models.base_models.knn import KNNModel

def test_knn_train_and_predict():
    X_train = np.array([[1], [2], [3], [4], [5], [6]])  # 6 samples
    y_train = np.array([1, 2, 3, 4, 5, 6])
    X_test = np.array([[7]])
    model = KNNModel(n_neighbors=3)  # Specify fewer neighbors
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    assert prediction.shape == (1,)
    assert isinstance(prediction[0], (int, float, np.integer, np.floating))
