import cv2
import os

# Load Haar Cascade (AMAN)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    raise IOError("Haar Cascade gagal dimuat")

# Buka kamera
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    raise IOError("Kamera tidak bisa dibuka")

person_name = "siti"
save_dir = f"dataset/{person_name}"
os.makedirs(save_dir, exist_ok=True)

count = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("Frame tidak terbaca")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))
        cv2.imwrite(f"{save_dir}/{count}.jpg", face)
        count += 1

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Count: {count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Capture", frame)

    if cv2.waitKey(1) & 0xFF == ord("q") or count >= 30:
        break

cam.release()
cv2.destroyAllWindows()
