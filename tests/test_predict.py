import pandas as pd
import pickle

def test_prediction():
    # Load model
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Load new data
    new_data = pd.DataFrame({'feature1': [0.5], 'feature2': [1.5]})

    # Predict
    predictions = model.predict(new_data)
    assert len(predictions) == 1