from flask import Flask, request, render_template, jsonify
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import io
import base64
import os

app = Flask(__name__)

# Load trained model
MODEL_PATH = 'model/ndt_flaw_model.keras'
model = None

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully!")
    else:
        print("Model not found. Please train the model first.")

def preprocess_image(image_data):
    """
    Preprocess uploaded image for prediction
    Expected input: (480, 7168) grayscale image
    """
    # Convert to numpy array
    img = Image.open(io.BytesIO(image_data))
    
    # Convert to grayscale if needed
    if img.mode != 'L':
        img = img.convert('L')
    
    # Resize to expected dimensions if needed
    if img.size != (7168, 480):
        img = img.resize((7168, 480), Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Reshape for model input
    img_array = img_array.reshape(1, 480, 7168, 1)
    
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({
            'error': 'Model not loaded. Please train the model first.'
        }), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read and preprocess image
        image_data = file.read()
        processed_image = preprocess_image(image_data)
        
        # Make prediction
        prediction = model.predict(processed_image, verbose=0)
        confidence = float(prediction[0][0])
        
        # Classify
        is_flaw = confidence >= 0.5
        
        result = {
            'prediction': 'FLAW DETECTED' if is_flaw else 'NO FLAW',
            'confidence': f"{confidence * 100:.2f}%" if is_flaw else f"{(1 - confidence) * 100:.2f}%",
            'raw_score': confidence,
            'class': 'flaw' if is_flaw else 'no-flaw'
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)