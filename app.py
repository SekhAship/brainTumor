from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Flask app setup
app = Flask(__name__)

# Config
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = 'model\\model2brainTumor.h5'  # Place your .h5 model inside a folder named 'model'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload folder if not exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load model once at startup
model = load_model(MODEL_PATH)

# Class mapping
CATEGORY = {
    0: "pituitary",
    1: "notumor",
    2: "meningioma",
    3: "glioma"
}


# Preprocessing function
def img_process(file_path):
    img = cv2.imread(file_path)
    img = cv2.resize(img, (256, 256))
    img = img / 255.0
    return img


# Prediction function
def predict_tumor(file_path, model):
    img = img_process(file_path)
    img = np.expand_dims(img, axis=0)  # batch dimension

    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = float(prediction[0][predicted_class]) * 100  # Explicit cast

    return {
        "class": CATEGORY[predicted_class],
        "confidence": round(confidence, 2)  # Now it's safe
    }


# Route to serve uploaded images (optional for displaying in frontend)
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# Home route
@app.route('/')
def index():
    return render_template('index.html')  # optional if using frontend


# Prediction route
@app.route('/predict', methods=['POST'])
def find_tumor():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        try:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            print(f"File saved to {file_path}")

            prediction = predict_tumor(file_path, model)

            # Clean up uploaded image after prediction
            os.remove(file_path)

            return jsonify(prediction), 200
        except Exception as e:
            print(f"Error processing file: {e}")
            return jsonify({"error": str(e)}), 500


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
