import os

def test_model_exists():
    assert os.path.exists('knn_model.joblib')
