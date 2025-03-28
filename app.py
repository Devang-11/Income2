# app.py
from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the label encoders
with open('label_encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

# Expected feature order
FEATURES = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
            'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
            'hours-per-week', 'native-country']

# Categorical features that need encoding
CATEGORICAL_FEATURES = [col for col in FEATURES if col in encoders]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Validate input
        if not all(feature in data for feature in FEATURES):
            return jsonify({'error': 'Missing required features'}), 400
        
        # Create a dictionary to hold processed input
        processed_data = {}
        
        # Process each feature
        for feature in FEATURES:
            if feature in CATEGORICAL_FEATURES:
                # Transform English input to numeric using the saved encoder
                try:
                    processed_data[feature] = encoders[feature].transform([data[feature]])[0]
                except ValueError:
                    return jsonify({'error': f'Invalid value for {feature}: {data[feature]}'}), 400
            else:
                # Numeric features: convert to float/int
                processed_data[feature] = pd.to_numeric(data[feature], errors='coerce')
        
        # Check for invalid numeric values
        if any(pd.isna(val) for val in processed_data.values()):
            return jsonify({'error': 'Invalid numeric input values'}), 400
        
        # Create DataFrame from processed input
        input_data = pd.DataFrame([processed_data], columns=FEATURES)
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0].max()
        
        # Return result
        return jsonify({
            'prediction': int(prediction),
            'probability': float(probability),
            'income': '>50K' if prediction == 1 else '<=50K'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    return jsonify({
        'message': 'Welcome to the Income Prediction API',
        'endpoint': '/predict',
        'method': 'POST',
        'expected_features': FEATURES,
        'note': 'Use English text for categorical features (e.g., "workclass": "Private")'
    })

if __name__ == '__main__':
    from waitress import serve
    serve(app, host="0.0.0.0", port=8000)
