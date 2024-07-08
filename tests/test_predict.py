import pandas as pd
import joblib

def test_prediction():
    # Load model
    LR_model = joblib.load('model.pkl')
    
    # Load new data
    new_data = pd.DataFrame({'total_room': [5], 'total_bed': [10]})

    # Predict
    predictions = LR_model.predict(new_data)
    assert len(predictions) == 1