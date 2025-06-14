import gdown
from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import requests

# Download model from Google Drive if not present
def download_from_gdrive(file_id, dest_path):
    print("Downloading model from Google Drive...")
    URL = "https://drive.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)

    def get_confirm_token(resp):
        for key, value in resp.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    token = get_confirm_token(response)
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)
    print("Model downloaded!")

# ✅ Real FILE ID from your Google Drive link

FILE_ID = "1QtR7oLr2X2sveFHlldCDWbQMmfnYBy2H"
MODEL_PATH = os.path.join("model", "model2brainTumor.h5")

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive using gdown...")
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    gdown.download(url, MODEL_PATH, fuzzy=True, quiet=False)
    print("Download complete.")
    
if os.path.getsize(MODEL_PATH) < 1024 * 100:
    raise ValueError("Downloaded model file seems too small. Likely not a valid model.")


# Flask app setup
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ✅ Load model once
model = load_model(MODEL_PATH)

# Class mapping
CATEGORY = {
    0: "pituitary",
    1: "notumor",
    2: "meningioma",
    3: "glioma"
}

# Preprocessing
def img_process(file_path):
    img = cv2.imread(file_path)
    img = cv2.resize(img, (256, 256))
    img = img / 255.0
    return img

# Prediction
def predict_tumor(file_path, model):
    img = img_process(file_path)
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = float(prediction[0][predicted_class]) * 100
    return {
        "class": CATEGORY[predicted_class],
        "confidence": round(confidence, 2)
    }

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/')
def index():
    return render_template('index.html')  # Optional frontend

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
            os.remove(file_path)

            return jsonify(prediction), 200
        except Exception as e:
            print(f"Error processing file: {e}")
            return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
