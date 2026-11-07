import cv2
import os
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()

detector = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

faces = []
labels = []

names = sorted(os.listdir("dataset"))

for label, name in enumerate(names):

    folder = os.path.join("dataset", name)

    if not os.path.isdir(folder):
        continue

    for file in os.listdir(folder):

        path = os.path.join(folder, file)

        img = cv2.imread(path)

        if img is None:
            continue

        gray = cv2.cvtColor(
            img, cv2.COLOR_BGR2GRAY
        )

        detected = detector.detectMultiScale(
            gray, 1.3, 5
        )

        for (x, y, w, h) in detected:

            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))

            faces.append(face)
            labels.append(label)

    print(f"Verwerkt: {name}")

if not faces:
    raise ValueError("Geen gezichten gevonden!")

recognizer.train(
    faces,
    np.array(labels)
)

os.makedirs("trainer", exist_ok=True)

recognizer.save("trainer/trainer.yml")

with open("trainer/names.txt", "w") as f:
    f.write("\n".join(names))

print("Training voltooid!")