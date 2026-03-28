

from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import os

app = Flask(__name__)

# Global variables to store model and scaler
model = None
scaler = None

def load_model():
    """
    Load the trained model and scaler from disk.
    """
    global model, scaler
    
    try:
       
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, 'model', 'fraud_model.pkl')
        scaler_path = os.path.join(script_dir, 'model', 'scaler.pkl')
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        print("Model and scaler loaded successfully!")
        return True
    except FileNotFoundError:
        print("ERROR: Model files not found!")
        print("Please run: python train.py")
        return False

def validate_input(data):
    """
    Validate user input.
    
    Args:
        data: Dictionary with transaction details
    
    Returns:
        (is_valid, error_message)
    """
    required_fields = ['time', 'amount', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
    
    for field in required_fields:
        if field not in data:
            return False, f"Missing field: {field}"
        
        try:
            value = float(data[field])
        except (ValueError, TypeError):
            return False, f"Invalid value for {field}. Must be a number."
    
    
    time_val = float(data['time'])
    amount_val = float(data['amount'])
    
    if not (0 <= time_val <= 48):
        return False, "Time must be between 0 and 48 hours"
    
    if amount_val < 0:
        return False, "Amount must be non-negative"
    
    return True, None

def predict_fraud(features):
    """
    Make a fraud prediction using the trained model.
    
    Args:
        features: List of feature values
    
    Returns:
        Dictionary with prediction, probability, and confidence
    """
    
    features_scaled = scaler.transform([features])
    
    
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    
    
    fraud_probability = probability[1]  # Probability of fraud
    confidence = max(probability) * 100
    
    return {
        'is_fraud': bool(prediction),
        'fraud_probability': float(fraud_probability),
        'legitimate_probability': float(probability[0]),
        'confidence': float(confidence),
        'prediction_text': 'Fraudulent Transaction' if prediction else 'Legitimate Transaction'
    }

@app.route('/')
def home():
    """
    Home page route.
    """
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    Prediction page route.
    Displays the form for user input.
    """
    if request.method == 'POST':
        # Get JSON data from request
        data = request.get_json()
        
        
        is_valid, error_msg = validate_input(data)
        
        if not is_valid:
            return jsonify({'success': False, 'error': error_msg}), 400
        
        try:
            # Extract features
            features = [
                float(data['time']),
                float(data['amount']),
                float(data['v1']),
                float(data['v2']),
                float(data['v3']),
                float(data['v4']),
                float(data['v5']),
                float(data['v6'])
            ]
            
            
            result = predict_fraud(features)
            
            return jsonify({
                'success': True,
                'prediction': result['prediction_text'],
                'is_fraud': result['is_fraud'],
                'fraud_probability': round(result['fraud_probability'], 4),
                'legitimate_probability': round(result['legitimate_probability'], 4),
                'confidence': round(result['confidence'], 2),
                'input_data': {
                    'time': float(data['time']),
                    'amount': float(data['amount'])
                }
            })
        
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    return render_template('predict.html')

@app.route('/api/sample-data', methods=['GET'])
def get_sample_data():
    """
    API endpoint to provide sample data for testing.
    """
    samples = {
        'legitimate': {
            'time': 10,
            'amount': 50,
            'v1': 0.5,
            'v2': -0.8,
            'v3': 1.2,
            'v4': -0.3,
            'v5': 0.1,
            'v6': -0.5
        },
        'fraudulent': {
            'time': 2,
            'amount': 800,
            'v1': 5.5,
            'v2': -5.2,
            'v3': 3.8,
            'v4': -4.1,
            'v5': 1.9,
            'v6': -2.7
        }
    }
    return jsonify(samples)

@app.route('/about')
def about():
    """
    About/Info page explaining fraud detection.
    """
    return render_template('about.html')

@app.route('/health')
def health():
    """
    Health check endpoint.
    """
    model_loaded = model is not None and scaler is not None
    return jsonify({
        'status': 'ok' if model_loaded else 'error',
        'model_loaded': model_loaded
    })

@app.errorhandler(404)
def not_found(error):
    """
    Handle 404 errors.
    """
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(error):
    """
    Handle 500 errors.
    """
    return render_template('500.html'), 500

if __name__ == '__main__':
    
    if not load_model():
        print("\nWARNING: App starting without model!")
        print("Some features may not work correctly.")
    
    
    print("\n" + "="*60)
    print("CREDIT CARD FRAUD DETECTION - FLASK APP")
    print("="*60)
    print("Starting Flask development server...")
    print("Open your browser and go to http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    app.run(debug=True, host='127.0.0.1', port=5000)
