import os
import numpy as np
from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense as KerasDense
from utils.preprocess import preprocess_image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

class CompatDense(KerasDense):
    def __init__(self, *args, quantization_config=None, **kwargs):
        super().__init__(*args, **kwargs)

# Load model once at startup
MODEL_PATH = 'models/alphabet_cnn.h5'
model = load_model(MODEL_PATH, custom_objects={'Dense': CompatDense})
print("[INFO] Model loaded for inference.")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded.", 400

    file = request.files['file']

    if file.filename == '' or not allowed_file(file.filename):
        return "Invalid file type. Please upload a JPG or PNG image.", 400

    # Save uploaded file safely
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Preprocess and predict
    processed = preprocess_image(filepath)
    prediction = model.predict(processed, verbose=0)
    
    # Get highest probability index
    predicted_index = np.argmax(prediction)
    
    # Map index 0-25 to ASCII 'A'-'Z'
    predicted_char = chr(predicted_index + ord('A'))
    confidence = float(np.max(prediction)) * 100

    return render_template('result.html',
                           character=predicted_char,
                           confidence=f"{confidence:.2f}",
                           image_path=filepath)

if __name__ == '__main__':
    app.run(debug=False)
