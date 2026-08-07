import os
import cv2
import json
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load model
model = load_model("model.h5")

# Load labels
with open("labels.json", "r") as f:
    label_map = json.load(f)

labels = list(label_map.keys())

IMG_SIZE = 128

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    image_path = None

    if request.method == "POST":
        file = request.files["file"]

        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            image_path = filepath

            # Preprocess image
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.equalizeHist(img)
            img = img.reshape(1, 128, 128, 1) / 255.0

            # Predict
            pred = model.predict(img)
            index = np.argmax(pred)

            result = labels[index]
            confidence = float(np.max(pred)) * 100

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        image_path=image_path
    )

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)