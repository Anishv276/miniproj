import cv2
import numpy as np
import os
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# =========================
# LOAD MODEL
# =========================
print("Loading model...")
model = load_model("model.h5")

# =========================
# LOAD LABELS
# =========================
with open("labels.json", "r") as f:
    label_map = json.load(f)

# Keep order consistent
labels = sorted(label_map.keys())

# =========================
# LOAD DATASET
# =========================
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

        # SAME preprocessing as training
        img = cv2.resize(img, (128,128))
        img = cv2.equalizeHist(img)

        X.append(img)
        y.append(label_map[label])

# =========================
# CONVERT DATA
# =========================
X = np.array(X).reshape(-1,128,128,1) / 255.0
y = np.array(y)

print("Total samples:", len(X))

# =========================
# TRAIN-TEST SPLIT (SAME AS TRAINING)
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print("Test samples:", len(X_test))

# =========================
# PREDICT
# =========================
print("Predicting...")

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

y_true = y_test

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_true, y_pred_classes)

print("\nConfusion Matrix (Proper Format):\n")

# Print header
print("      " + " ".join(f"{label:>6}" for label in labels))

# Print rows with labels
for i, row in enumerate(cm):
    print(f"{labels[i]:>6} " + " ".join(f"{num:6d}" for num in row))

# =========================
# MODEL EVALUATION (LOSS + ACCURACY)
# =========================
print("\nEvaluating model on test data...")

y_test_cat = to_categorical(y_test, num_classes=len(labels))

test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)

print(f"\nTest Loss: {test_loss:.5f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# =========================
# METRICS
# =========================
accuracy = accuracy_score(y_true, y_pred_classes)
precision = precision_score(y_true, y_pred_classes, average='weighted')
recall = recall_score(y_true, y_pred_classes, average='weighted')
f1 = f1_score(y_true, y_pred_classes, average='weighted')

print(f"\nAccuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F-Measure: {f1:.3f}")