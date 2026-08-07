import os

DATASET_PATH = "fingerprint samples/dataset_blood_group"

if not os.path.exists(DATASET_PATH):
    print("Dataset path not found!")
    exit()

labels = sorted(os.listdir(DATASET_PATH))

print("Dataset distribution:\n")

total = 0

for label in labels:
    folder = os.path.join(DATASET_PATH, label)
    count = len(os.listdir(folder))
    total += count
    print(f"{label}: {count} images")

print("\nTotal images:", total)