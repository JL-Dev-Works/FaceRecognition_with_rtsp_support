
import cv2
import os
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()

detector = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

faces = []
labels = []

names = sorted([
    name for name in os.listdir("dataset")
    if os.path.isdir(os.path.join("dataset", name))
])


def augment_face(face):

    variants = []

    # Origineel
    variants.append(face)

    # Verschillende helderheden
    for alpha, beta in [
        (0.7, -20),
        (0.85, 15),
        (1.2, 20),
        (1.4, 35)
    ]:
        adjusted = cv2.convertScaleAbs(
            face,
            alpha=alpha,
            beta=beta
        )
        variants.append(adjusted)

    # Verschillende contrasten
    for alpha in [0.7, 1.3]:
        contrast = cv2.convertScaleAbs(
            face,
            alpha=alpha,
            beta=0
        )
        variants.append(contrast)

    # Gaussian blur
    for sigma in [1, 2]:
        blurred = cv2.GaussianBlur(
            face,
            (5, 5),
            sigma
        )
        variants.append(blurred)

    # Motion blur
    kernel = np.zeros((9, 9), dtype=np.float32)
    kernel[4, :] = 1
    kernel /= kernel.sum()

    motion = cv2.filter2D(
        face,
        -1,
        kernel
    )
    variants.append(motion)

    # Digitale ruis
    noise = np.random.normal(
        0, 12, face.shape
    )

    noisy = np.clip(
        face.astype(np.float32) + noise,
        0, 255
    ).astype(np.uint8)

    variants.append(noisy)

    # Lage resolutie
    small = cv2.resize(
        face,
        (40, 40),
        interpolation=cv2.INTER_AREA
    )

    low_resolution = cv2.resize(
        small,
        (200, 200),
        interpolation=cv2.INTER_CUBIC
    )

    variants.append(low_resolution)

    return variants


for label, name in enumerate(names):

    folder = os.path.join("dataset", name)

    processed = 0

    for file in os.listdir(folder):

        path = os.path.join(folder, file)

        img = cv2.imread(path)

        if img is None:
            continue

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

        detected = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(50, 50)
        )

        for (x, y, w, h) in detected:

            face = gray[y:y+h, x:x+w]

            face = cv2.resize(
                face,
                (200, 200),
                interpolation=cv2.INTER_CUBIC
            )

            # Data augmentation
            variants = augment_face(face)

            for variant in variants:

                faces.append(variant)
                labels.append(label)

            processed += 1

    print(
        f"{name}: {processed} gezichten verwerkt"
    )


if not faces:
    raise ValueError("Geen gezichten gevonden!")


# Model trainen
recognizer.train(
    faces,
    np.array(labels, dtype=np.int32)
)


# Opslaan
os.makedirs("trainer", exist_ok=True)

recognizer.save(
    "trainer/trainer.yml"
)

with open("trainer/names.txt", "w") as f:
    f.write("\n".join(names))


print("\nTraining voltooid!")
print(f"Totaal trainingsbeelden: {len(faces)}")
print(f"Aantal personen: {len(names)}")
