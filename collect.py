import cv2
import os

name = input("Naam: ")

path = f"dataset/{name}"
os.makedirs(path, exist_ok=True)

camera = cv2.VideoCapture(0)

face_detector = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

count = 0

while True:
    ret, frame = camera.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray, 1.3, 5
    )

    for (x, y, w, h) in faces:
        count += 1

        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))

        cv2.imwrite(
            f"{path}/{count}.jpg",
            face
        )

        cv2.rectangle(
            frame, (x, y),
            (x+w, y+h), (0, 255, 0), 2
        )

    cv2.imshow("Collect Faces", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()