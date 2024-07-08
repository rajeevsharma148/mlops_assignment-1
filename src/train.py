import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load data
data = pd.read_csv('data/data.csv')
X = data[['feature1', 'feature2']]
y = data['target']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
with open('model/model.pkl', 'wb') as f:
    pickle.dump(model, f)