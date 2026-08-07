from flask import Flask, request, jsonify
import cv2
import numpy as np
import json
from tensorflow.keras.models import load_model
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = load_model("model.h5")

with open("labels.json", "r") as f:
    label_map = json.load(f)

labels = list(label_map.keys())

@app.route("/predict", methods=["POST"])
def predict():
    #  Check file exists
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    # Check empty file
    if file.filename == "":
        return jsonify({"error": "Empty file"}), 400

    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    # Critical validation
    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    try:
        img = cv2.resize(img, (128,128))
        img = cv2.equalizeHist(img)
        img = img.reshape(1,128,128,1)/255.0
    except:
        return jsonify({"error": "Image processing failed"}), 400

    pred = model.predict(img)[0]
    confidence = float(np.max(pred) * 100)

    # Confidence filter
    if confidence < 70:
        return jsonify({
            "error": "Low confidence. Upload a proper fingerprint."
        }), 400

    index = np.argmax(pred)

    return jsonify({
        "predicted": labels[index],
        "confidence": confidence,
        "labels": labels,
        "probabilities": (pred * 100).tolist()
    })

if __name__ == "__main__":
    app.run(port=5000)