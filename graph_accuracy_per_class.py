import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
DATASET_PATH = os.path.join("fingerprint samples", "dataset_blood_group")
IMG_SIZE = 128
# Load labels
with open("labels.json", "r") as f:
    label_map = json.load(f)
labels = list(label_map.keys())
# Load model
model = load_model("model.h5")
# Load data
X, y = [], []
for label in labels:
    folder = os.path.join(DATASET_PATH, label)
    for img_name in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, img_name), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
        img = np.expand_dims(img, axis=-1)
        X.append(img)
        y.append(label_map[label])
X = np.array(X)
y = np.array(y)
# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
# Predict
y_pred = np.argmax(model.predict(X_test), axis=1)
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
# Safe accuracy calculation
row_sums = cm.sum(axis=1)
accuracy = []
for i in range(len(row_sums)):
    if row_sums[i] == 0:
        accuracy.append(0)
    else:
        accuracy.append(cm[i][i] / row_sums[i])
# Plot
plt.figure(figsize=(8, 5))
plt.bar(labels, accuracy)
plt.title("Accuracy per Class")
plt.xlabel("Blood Group")
plt.ylabel("Accuracy")
# Save instead of show
plt.savefig("accuracy_per_class.png")
plt.close()
print("Saved as accuracy_per_class.png")