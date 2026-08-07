import cv2
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
print("Loading model...")
model = load_model("model.h5")
with open("labels.json", "r") as f:
    label_map = json.load(f)
labels = sorted(label_map.keys())
DATASET_PATH = "fingerprint samples/dataset_blood_group"
X = []
y = []
print("Loading dataset...")
for label in labels:
    folder_path = os.path.join(DATASET_PATH, label)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (128,128))
        img = cv2.equalizeHist(img)

        X.append(img)
        y.append(label_map[label])
X = np.array(X).reshape(-1,128,128,1) / 255.0
y = np.array(y)
print("Total samples:", len(X))
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)
print("Test samples:", len(X_test))
print("Predicting on test data...")
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(10,8))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='viridis',
    xticklabels=labels,
    yticklabels=labels
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Test Data)")
plt.savefig("confusion_matrix_test.png", dpi=300, bbox_inches='tight')
print("Confusion matrix saved as confusion_matrix_test.png")
plt.show()
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_classes, target_names=labels))