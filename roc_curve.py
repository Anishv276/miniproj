# graph_roc_curve.py

import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

DATASET_PATH = "fingerprint samples\dataset_blood_group"
IMG_SIZE = 128

with open("labels.json", "r") as f:
    label_map = json.load(f)

labels = list(label_map.keys())
num_classes = len(labels)

model = load_model("model.h5")

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

split = int(0.8 * len(X))
X_test = X[split:]
y_test = y[split:]

y_pred_probs = model.predict(X_test)

y_test_bin = label_binarize(y_test, classes=list(range(num_classes)))

for i in range(num_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{labels[i]} (AUC={roc_auc:.2f})")

plt.plot([0, 1], [0, 1])
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve")
plt.legend()
plt.savefig("roc.png")
plt.close()