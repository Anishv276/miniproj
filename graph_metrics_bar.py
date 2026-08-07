import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
DATASET_PATH = "fingerprint samples/dataset_blood_group"
IMG_SIZE = 128
with open("labels.json", "r") as f:
    label_map = json.load(f)

labels = list(label_map.keys())
model = load_model("model.h5")
X, y = [], []
for label in labels:
    folder = os.path.join(DATASET_PATH, label)
    if not os.path.exists(folder):
        print(f"WARNING: Folder not found -> {folder}")
        continue
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0
        img = np.expand_dims(img, axis=-1)
        X.append(img)
        y.append(label_map[label])
X = np.array(X)
y = np.array(y)
print("Total samples:", len(X))
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
y_pred = np.argmax(model.predict(X_test), axis=1)
report = classification_report(
    y_test,
    y_pred,
    labels=list(range(len(labels))),
    target_names=labels,
    output_dict=True,
    zero_division=0
)
precision = [report[label]["precision"] for label in labels]
recall = [report[label]["recall"] for label in labels]
f1 = [report[label]["f1-score"] for label in labels]
x = np.arange(len(labels))
plt.figure(figsize=(10, 5))
plt.bar(x - 0.2, precision, width=0.2, label="Precision")
plt.bar(x, recall, width=0.2, label="Recall")
plt.bar(x + 0.2, f1, width=0.2, label="F1-score")
plt.xticks(x, labels)
plt.xlabel("Blood Group")
plt.ylabel("Score")
plt.title("Precision / Recall / F1 Score")
plt.legend()
plt.savefig("metrics_bar.png")
plt.close()
print("Graph saved as metrics_bar.png")
from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred)
print(report)