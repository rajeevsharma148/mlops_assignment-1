import pandas as pd
import joblib

# Load model
LR_model = joblib.load('model.pkl')

# Load new data
new_data = pd.DataFrame({'feature1': [0.5], 'feature2': [1.5]})

# Predict
predictions = LR_model.predict(new_data)
print(predictions)