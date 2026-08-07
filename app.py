import os
import cv2
import json
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
try:
    from tensorflow.keras.models import load_model
except ImportError:
    print("Tensorflow not imported.")
    import sys; sys.exit(1)
app = Flask(__name__)
CORS(app)
print("Initializing AI Model Backend...")
model = load_model("model.h5")
print("Model loaded successfully.")
with open("labels.json", "r") as f:
    label_map = json.load(f)
labels = list(label_map.keys())
IMG_SIZE = 128
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file parameter found"}), 400
    file = request.files["file"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    try:
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return jsonify({"error": "Unreadable image file"}), 400
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.equalizeHist(img)
        img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1) / 255.0
        pred = model.predict(img)
        index = int(np.argmax(pred))
        result = labels[index]
        confidence = float(np.max(pred)) * 100
        probabilities = [float(p) * 100 for p in pred[0]]
        os.remove(filepath)
        return jsonify({
            "predicted": result,
            "confidence": confidence,
            "probabilities": probabilities,
            "labels": labels
        })
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        print("Error during predict:", str(e))
        return jsonify({"error": "Internal Model Error"}), 500
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=False)