import cv2
import json
import csv
import os
from datetime import datetime

# folder script / project
BASE_DIR = os.path.join(os.getcwd(), "face_recognition") 
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# folder absensi
ABSEN_DIR = os.path.join(BASE_DIR, "absensi")
os.makedirs(ABSEN_DIR, exist_ok=True)  

# paths file assets
cascade_path = os.path.join(ASSETS_DIR, 'haarcascade_frontalface_default.xml')
trainer_path = os.path.join(ASSETS_DIR, 'face_model.xml')
labels_file = os.path.join(ASSETS_DIR, 'labels.json')

# file CSV otomatis sesuai tanggal hari ini, masuk folder absensi
today_str = datetime.now().strftime('%Y-%m-%d')
attendance_file = os.path.join(ABSEN_DIR, f'Attendance_{today_str}.csv')

# load model & label
face_cascade = cv2.CascadeClassifier(cascade_path)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(trainer_path)

with open(labels_file, 'r') as f:
    raw_map = json.load(f)
    id_to_name = {int(k): v for k, v in raw_map.items()}

# buat file CSV
if not os.path.exists(attendance_file):
    with open(attendance_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Id', 'Name', 'Date', 'Time', 'Status'])

# simpan absen di memory agar tidak double entry
absen_today = set()

def mark_attendance(Id):
    if Id in absen_today:
        return  # langsung return kalau sudah absen
    name = id_to_name.get(Id, "Unknown")
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    time_str = now.strftime('%H:%M:%S')

    with open(attendance_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([Id, name, date_str, time_str, 'Present'])
    absen_today.add(Id)
    print(f"Absensi Dicatat: {name} pada {time_str}")

# mulai webcam
cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

while True:
    ret, img = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(int(minW), int(minH)))

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        Id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        if confidence < 100:
            name = id_to_name.get(Id, "Unknown")
            confidence_level = f"{round(100 - confidence)}%"
            mark_attendance(Id)
        else:
            name = "Unknown"
            confidence_level = f"{round(100 - confidence)}%"

        cv2.putText(img, name, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, confidence_level, (x + 5, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)

    cv2.imshow("Attendance System", img)
    if cv2.waitKey(10) & 0xff == 27:  # ESC untuk keluar
        break

cam.release()
cv2.destroyAllWindows()
