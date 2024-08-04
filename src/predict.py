from flask import Flask, request, jsonify
import joblib
import numpy as np
# Initialize Flask application  
app = Flask(__name__)

knn_model = joblib.load('knn_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        features = np.array(data['features'], dtype=float).reshape(1, -1)
        prediction = knn_model.predict(features)
        response = {
            'prediction': int(prediction[0])
        }
        return jsonify(response)
    except ValueError:
        return jsonify({"error": "Invalid input data: must be numeric"}), 400  # Return 400 for invalid input
    except Exception as e:
        return jsonify({"error": str(e)}), 500  # Return 500 for server error
    
if __name__ == '__main__':
    app.run(debug=True)