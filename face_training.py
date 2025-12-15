import cv2
import os
import numpy as np
import json 

def load_dataset(dataset_path="dataset"):
    faces = []
    labels = []
    label_ids = {}  # mapping label → nama orang
    current_label = 0

    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_path):
            continue

        label_ids[current_label] = person_name

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            faces.append(img)
            labels.append(current_label)

        current_label += 1

    return faces, labels, label_ids

# paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "..", "dataset")
DATASET_DIR = os.path.abspath(DATASET_DIR)
print("Path dataset:", DATASET_DIR)

ASSETS_DIR = os.path.join(BASE_DIR, "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

# load dataset & train
faces, labels, label_ids = load_dataset(DATASET_DIR)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))

# simpan model & labels
recognizer.save(os.path.join(ASSETS_DIR, "face_model.xml"))

with open(os.path.join(ASSETS_DIR, "labels.json"), "w") as f:
    json.dump({str(k): v for k, v in label_ids.items()}, f)

print("Training selesai! Model tersimpan di assets/face_model.xml")
