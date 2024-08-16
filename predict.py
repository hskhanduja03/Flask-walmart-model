from flask import Flask, request, jsonify
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from flask_cors import CORS
from huggingface_hub import hf_hub_download

app = Flask(__name__)
CORS(app)

# Load the first model (retail.keras)
model_keras = load_model('retail.keras')

# Load the second model (random_forest_model) from Hugging Face
token = "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"  # Replace with your actual token
model_path = hf_hub_download(
    repo_id='arnab12345678/item_quantity_finder',
    filename='random_forest_model.pkl',
    token=token
)
model_rf = joblib.load(model_path)

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
        predictions = model_keras.predict(input_data)
        return jsonify({'predictions': predictions.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_rf', methods=['POST'])
def predict_rf():
    data = request.get_json()  # Expecting JSON data
    if not data:
        return jsonify({'error': 'No input data provided'}), 400

    try:
        # Extract features from the incoming JSON
        features = [
            data[0],  # Assuming the order matches the model's input features
            data[1],
            data[2],
            data[3],
            data[4],
            data[5],
            data[6],
            data[7],
            data[8],
            data[9]
        ]
        
        # Convert features to a numpy array and reshape for prediction
        input_data = np.array([features])
        predictions = model_rf.predict(input_data)
        
        return jsonify({'predictions': predictions.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test', methods=['GET'])
def test():
    return jsonify({'message': 'Hello from server'})

if __name__ == '__main__':
    app.run(port=8000, host='0.0.0.0')
