import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import base64
import io
import os


MODEL_PATH = "densenet_model.keras"
CASCADE_PATH = "haarcascade_frontalface_default.xml"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file not found: densenet_model.keras")

if not os.path.exists(CASCADE_PATH):
    raise FileNotFoundError("Haarcascade file not found: haarcascade_frontalface_default.xml")

model = tf.keras.models.load_model(MODEL_PATH)
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# Helper functions

def detect_and_crop_face(image_pil: Image.Image):
    img = np.array(image_pil)
    img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    if len(faces) == 0:
        return None, img_cv

    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    face = img_cv[y:y + h, x:x + w]
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    return Image.fromarray(face_rgb), img_cv


def predict_skin_tone(image_pil: Image.Image):
    img = np.array(image_pil)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img_tensor = np.expand_dims(img, axis=0)
    preds = model.predict(img_tensor)
    pred_class = np.argmax(preds)
    confidence = float(np.max(preds) * 100)
    return pred_class, confidence


def image_to_base64(image_pil: Image.Image):
    buffered = io.BytesIO()
    image_pil.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


