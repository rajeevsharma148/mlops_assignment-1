import sys
import os

# Insert the parent directory of 'src' into sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.predict import app
import unittest
import json

class TestPredictEndpoint(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_predict_endpoint(self):
        # Prepare a sample request data
        request_data = {
            'features': [1.0, 2.0, 3.0, 4.0]  # Example feature values
        }

        # Convert dictionary to JSON string
        json_data = json.dumps(request_data)

        # Make POST request to /predict endpoint
        response = self.app.post('/predict', data=json_data, content_type='application/json')

        # Check if status code is 200 OK
        self.assertEqual(response.status_code, 200)

        # Check if response is in JSON format
        self.assertEqual(response.content_type, 'application/json')

        # Decode JSON response data
        response_data = json.loads(response.data.decode('utf-8'))

        # Verify the structure of the response
        self.assertIn('prediction', response_data)
        self.assertIsInstance(response_data['prediction'], int)

        # Add more specific assertions based on your prediction logic and expected outputs
        # Example:
        # self.assertEqual(response_data['prediction'], expected_prediction_value)

if __name__ == '__main__':
    unittest.main()