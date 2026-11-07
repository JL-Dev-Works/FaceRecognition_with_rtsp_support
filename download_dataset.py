
from sklearn.datasets import fetch_lfw_people
import os
import cv2
import numpy as np

print("LFW dataset ophalen...", flush=True)

# Dataset laden
lfw = fetch_lfw_people(
    min_faces_per_person=10,
    color=False,
    resize=1.0
)

print("Dataset geladen!", flush=True)
print("Aantal beschikbare personen:", len(lfw.target_names))

# Instellingen
AANTAL_PERSONEN = 4
FOTOS_PER_PERSOON = 10

# Datasetmap aanmaken
os.makedirs("dataset", exist_ok=True)

# Eerste 4 personen selecteren
personen = lfw.target_names[:AANTAL_PERSONEN]

for person_id, person_name in enumerate(personen):

    folder = os.path.join("dataset", person_name)
    os.makedirs(folder, exist_ok=True)

    # Afbeeldingen van deze persoon zoeken
    indices = np.where(lfw.target == person_id)[0]

    for i, index in enumerate(indices[:FOTOS_PER_PERSOON]):

        # Afbeelding ophalen
        image = lfw.images[index]

        # Waarden omzetten naar 0-255
        image = cv2.normalize(
            image,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )

        image = image.astype(np.uint8)

        # Afbeelding opslaan
        path = os.path.join(folder, f"{i}.jpg")
        cv2.imwrite(path, image)

    print(f"{person_name}: {min(len(indices), FOTOS_PER_PERSOON)} foto's opgeslagen")

print("\nDataset succesvol opgeslagen!")
print(f"Totaal: {AANTAL_PERSONEN * FOTOS_PER_PERSOON} foto's")
