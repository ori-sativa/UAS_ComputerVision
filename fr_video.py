import cv2
import os
import json
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(os.path.join(BASE_DIR, "assets", "face_model.xml"))


with open(os.path.join(BASE_DIR, "assets", "labels.json"), "r") as f:
    label_ids = json.load(f)

# load Haar Cascade
xmlPath = os.path.join(BASE_DIR, "assets", "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(xmlPath)


# buka webcam (0 = default camera)
cap = cv2.VideoCapture(0)

absen = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # konversi ke grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # deteksi wajah
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    # faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # gambar kotak pada wajah
    for x, y, w, h in faces:
        face_roi = gray[y : y + h, x : x + w]
        label, confidence = recognizer.predict(face_roi)

        if confidence < 80:
            name = "Unknown"
        else:   
            name = label_ids.get(str(label), "Unknown")

        # update absen jika nama belum ada
        if name not in absen:
            absen[name] = datetime.now().strftime("%Y-%m-%d %H:%M")

        text = f"{name} ({confidence:.1f})"

        cv2.putText(
            frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # posisi checklist
    y0 = 30
    for nama, waktu in absen.items():
        cv2.putText(
            frame,
            f"\u2713 {nama} - {waktu}",
            (10, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        y0 += 30
    
    # tampilkan output
    cv2.imshow("Face recognition", frame)

    # tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
