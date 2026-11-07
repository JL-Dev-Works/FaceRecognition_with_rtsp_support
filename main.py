
import os
import cv2
import time
import json
from urllib.parse import quote
from dotenv import load_dotenv
from deepface import DeepFace



load_dotenv()

SOURCE_MODE = "rtsp"  # "video" of "rtsp"

VIDEO_PATH = "test_video.mp4"

ANALYSE_INTERVAL = 2
RECOGNITION_THRESHOLD = 70

CASCADE_PATH = "haarcascade_frontalface_default.xml"
MODEL_PATH = "trainer/trainer.yml"
NAMES_PATH = "trainer/names.txt"



def get_rtsp_url():
    host = os.getenv("RTSP_HOST")
    port = os.getenv("RTSP_PORT", "554")
    username = os.getenv("RTSP_USERNAME")
    password = os.getenv("RTSP_PASSWORD")
    stream = os.getenv("RTSP_STREAM", "Preview_01_sub")

    if not all([host, username, password]):
        raise RuntimeError(
            "RTSP-configuratie ontbreekt in het .env-bestand!"
        )

    username = quote(username, safe="")
    password = quote(password, safe="")

    return (
        f"rtsp://{username}:{password}@{host}:{port}/{stream}"
    )



for path in [CASCADE_PATH, MODEL_PATH, NAMES_PATH]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Bestand niet gevonden: {path}")

if SOURCE_MODE == "video" and not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(f"Video niet gevonden: {VIDEO_PATH}")



cascade = cv2.CascadeClassifier(CASCADE_PATH)

if cascade.empty():
    raise RuntimeError("Haar Cascade kon niet worden geladen!")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_PATH)

with open(NAMES_PATH, "r", encoding="utf-8") as file:
    names = [line.strip() for line in file if line.strip()]

print("Gezichtsdetectie geladen!")
print("LBPH herkenning geladen!")
print("Namen:", names)



if SOURCE_MODE == "video":
    camera = cv2.VideoCapture(VIDEO_PATH)
elif SOURCE_MODE == "rtsp":
    rtsp_url = get_rtsp_url()
    camera = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
else:
    raise ValueError("SOURCE_MODE moet 'video' of 'rtsp' zijn!")

if not camera.isOpened():
    raise RuntimeError("Kan de videobron niet openen!")

print("Videobron gestart!")
print("Face Analysis gestart. Druk op Q om te stoppen.")



laatste_analyse = 0

data = {
    "face_detected": False,
    "person": "Unknown",
    "recognized": False,
    "confidence": None,
    "age": None,
    "emotion": None,
    "status": "no_face"
}



try:
    while True:
        ret, frame = camera.read()

        if not ret:
            if SOURCE_MODE == "video":
                print("Video afgelopen. Opnieuw beginnen...")
                camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
                laatste_analyse = 0
                continue

            print("Kan geen beeld ontvangen!")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(60, 60)
        )

        if len(faces) > 0:
            x, y, w, h = max(
                faces,
                key=lambda face: face[2] * face[3]
            )

            face_gray = gray[y:y+h, x:x+w]
            face_color = frame[y:y+h, x:x+w]

            huidige_tijd = time.time()

            if huidige_tijd - laatste_analyse >= ANALYSE_INTERVAL:
                print("Gezicht analyseren...")

                # PERSOON HERKENNEN
                label, confidence = recognizer.predict(face_gray)

                recognized = (
                    0 <= label < len(names)
                    and confidence < RECOGNITION_THRESHOLD
                )

                person = names[label] if recognized else "Unknown"

                # DEEPFACE ANALYSE
                try:
                    result = DeepFace.analyze(
                        img_path=face_color,
                        actions=["age", "emotion"],
                        enforce_detection=False,
                        silent=True
                    )

                    if isinstance(result, list):
                        result = result[0]

                    age = int(round(result["age"]))
                    emotion = result["dominant_emotion"]

                except Exception as e:
                    print("DeepFace fout:", e)
                    age = None
                    emotion = None

                # JSON OPBOUWEN
                data = {
                    "face_detected": True,
                    "person": person,
                    "recognized": recognized,
                    "confidence": round(float(confidence), 2),
                    "age": age,
                    "emotion": emotion,
                    "status": (
                        "recognized" if recognized
                        else "unknown_person"
                    )
                }

                print("\n--- FACE ANALYSIS JSON ---")
                print(json.dumps(data, indent=4, ensure_ascii=False))
                print("--------------------------\n")

                laatste_analyse = huidige_tijd

            # WEERGAVE
            color = (
                (0, 255, 0)
                if data["recognized"]
                else (0, 0, 255)
            )

            cv2.rectangle(
                frame,
                (x, y),
                (x+w, y+h),
                color,
                2
            )

            cv2.putText(
                frame,
                f"Persoon: {data['person']}",
                (x, max(25, y-65)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

            cv2.putText(
                frame,
                f"Leeftijd: {data['age']}",
                (x, max(25, y-35)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

            cv2.putText(
                frame,
                f"Emotie: {data['emotion']}",
                (x, max(25, y-10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        else:
            data = {
                "face_detected": False,
                "person": "Unknown",
                "recognized": False,
                "confidence": None,
                "age": None,
                "emotion": None,
                "status": "no_face"
            }

        # STATUS
        cv2.putText(
            frame,
            data["status"],
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.imshow("Face Analysis", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()

print("Face Analysis afgesloten.")
