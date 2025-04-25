import numpy as np
from models.hybrid_model import HybridModel
from models.base_models.knn import KNNModel
from models.base_models.svr import SVRModel

def test_hybrid_train_and_predict():
    X_train = np.random.rand(10, 3)
    y_train = np.random.rand(10)
    X_test = np.random.rand(2, 3)

    # Create models dictionary to pass to HybridModel
    models = {
        'knn': KNNModel(),
        'svr': SVRModel()
    }

    model = HybridModel(models=models)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    assert prediction.shape == (2,)
