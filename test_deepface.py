
from deepface import DeepFace
import cv2

# Testafbeelding laden
image = cv2.imread("test.jpg")

if image is None:
    print("Afbeelding test.jpg niet gevonden!")
    exit()

# Analyse uitvoeren
result = DeepFace.analyze(
    img_path=image,
    actions=["age", "emotion"],
    enforce_detection=True
)

print("\n--- FACE ANALYSIS ---")
print("Leeftijd:", result[0]["age"])
print("Emotie:", result[0]["dominant_emotion"])
print("Emoties:", result[0]["emotion"])
