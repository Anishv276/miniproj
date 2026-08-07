# graph_pie_distribution.py

import os
import json
import matplotlib.pyplot as plt

DATASET_PATH = "fingerprint samples/dataset_blood_group"

with open("labels.json", "r") as f:
    label_map = json.load(f)

labels = list(label_map.keys())

counts = []

for label in labels:
    folder = os.path.join(DATASET_PATH, label)
    counts.append(len(os.listdir(folder)))

plt.pie(counts, labels=labels, autopct="%1.1f%%")
plt.title("Class Distribution")
plt.savefig("pie_chart.png")
plt.close()