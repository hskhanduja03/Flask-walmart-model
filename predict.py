from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = load_model('retail.keras')

@app.route('/')
def home():
    return 'Welcome to the Flask API!'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Expecting JSON data
    if not data:
        return jsonify({'error': 'No input data provided'}), 400
    
    try:
        input_data = np.array(data['features'])
        predictions = model.predict(input_data)
        return jsonify({'predictions': predictions.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test', methods=['GET'])
def test():
    return jsonify({'message': 'Hello from server'})

if __name__ == '__main__':
    app.run(port=8000, host='0.0.0.0')
