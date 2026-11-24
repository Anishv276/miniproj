#predict.py
import cv2
import numpy as np
import json
from tensorflow.keras.models import load_model

# Load model
print("Loading model...")
model = load_model("model.h5")

# Load labels
print("Loading labels...")
with open("labels.json", "r") as f:
    label_map = json.load(f)

labels = list(label_map.keys())

# Load image
img_path = "testAB+.png"   # change if needed
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("ERROR: Image not found:", img_path)
    exit()

# Preprocessing
img = cv2.resize(img, (128,128))
img = cv2.equalizeHist(img)
img = img.reshape(1,128,128,1) / 255.0

print("Predicting...")

pred = model.predict(img)

index = np.argmax(pred)
result = labels[index]
confidence = np.max(pred) * 100

# Confidence handling
if confidence < 70:
    print("Prediction uncertain. Try another image.")
else:
    print("Prediction is relatively confident.")

print("Predicted Blood Group:", result)
print(f"Confidence: {confidence:.2f}%")