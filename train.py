import os
import cv2
import numpy as np
import json
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Dataset path
DATASET_PATH = "fingerprint samples/dataset_blood_group"

print("Checking dataset path...")
if not os.path.exists(DATASET_PATH):
    print("ERROR: Dataset path not found!")
    exit()

# Load labels
labels = sorted(os.listdir(DATASET_PATH))
print("Labels:", labels)

# Label mapping
label_map = {label: i for i, label in enumerate(labels)}

# Save mapping
with open("labels.json", "w") as f:
    json.dump(label_map, f)

X = []
y = []

print("Loading images...")

# Load dataset
for label in labels:
    folder_path = os.path.join(DATASET_PATH, label)

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Preprocessing
        img = cv2.resize(img, (128, 128))
        img = cv2.equalizeHist(img)

        X.append(img)
        y.append(label_map[label])

print("Total images:", len(X))

# Convert to numpy
X = np.array(X).reshape(-1, 128, 128, 1) / 255.0
y = to_categorical(y, num_classes=len(labels))

# Compute class weights (FIXES IMBALANCE)
y_integers = np.argmax(y, axis=1)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_integers),
    y=y_integers
)
class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)

# 80-20 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1
)

datagen.fit(X_train)

print("Building model...")

# Improved CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,1)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),

    Dense(256, activation='relu'),
    Dropout(0.5),

    Dense(len(labels), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Early stopping
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

print("Training started...")

model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=25,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    class_weight=class_weights   # IMPORTANT FIX
)

# Evaluate
print("Evaluating on test data...")
loss, accuracy = model.evaluate(X_test, y_test)

print(f"Test Accuracy: {accuracy*100:.2f}%")

# Save model
model.save("model.h5")

print("Training complete")