import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow as tf
import cv2
import os

# ----------------------------
# ✅ Must be the first Streamlit command
# ----------------------------
st.set_page_config(page_title="Skin Tone Classifier", page_icon="🎨", layout="centered")

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_trained_model():
    model = load_model("densenet_model.keras")
    return model

model = load_trained_model()

# ----------------------------
# Load Haar Cascade for Face Detection
# ----------------------------
cascade_path = os.path.join(os.getcwd(), "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    raise IOError("Failed to load Haar cascade. Check the path.")

# ----------------------------
# Helper Function: Detect and Crop Face
# ----------------------------
def detect_and_crop_face(image_pil):
    img = np.array(image_pil)
    img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    if len(faces) == 0:
        return None, img

    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    face = img_cv[y:y+h, x:x+w]
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    # draw rectangle
    cv2.rectangle(img_cv, (x, y), (x+w, y+h), (0, 255, 0), 3)
    detected_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    return Image.fromarray(face_rgb), Image.fromarray(detected_img)

# ----------------------------
# Helper Function for Prediction
# ----------------------------
def predict_img(image):
    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img_tensor = np.expand_dims(img, axis=0)
    preds = model.predict(img_tensor)
    return preds, img_tensor

# ----------------------------
# Saliency Map
# ----------------------------
def get_saliency_map(model, img_tensor):
    img_tensor = tf.convert_to_tensor(img_tensor)
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        preds = model(img_tensor)
        top_class = tf.argmax(preds[0])
        top_class_score = preds[:, top_class]

    grads = tape.gradient(top_class_score, img_tensor)
    saliency = tf.reduce_max(tf.abs(grads), axis=-1)[0]
    saliency = (saliency - tf.reduce_min(saliency)) / (tf.reduce_max(saliency) - tf.reduce_min(saliency))
    saliency = saliency.numpy()
    return saliency

# ----------------------------
# 🎨 Color Recommendations
# ----------------------------
color_recommendations = {
    "very_light": {"description": "Soft pastel tones enhance gentle undertones beautifully.",
                   "colors": ["#F4C2C2", "#FFD1DC", "#E0BBE4", "#C1E1C1", "#B0E0E6"]},
    "light": {"description": "Light, airy shades like sky blue, peach, and lavender look flattering.",
              "colors": ["#A7C7E7", "#FFB6C1", "#FFFACD", "#E6E6FA", "#98FB98"]},
    "light_medium": {"description": "Warm and neutral tones such as coral, turquoise, and plum work perfectly.",
                     "colors": ["#FF7F50", "#AFEEEE", "#D8BFD8", "#FFDAB9", "#F5DEB3"]},
    "medium": {"description": "Earthy colors like olive, rust, and mustard enhance your natural glow.",
               "colors": ["#808000", "#B7410E", "#DAA520", "#F4A460", "#CD853F"]},
    "medium_deep": {"description": "Rich jewel tones — emerald, navy, burgundy — make you stand out.",
                    "colors": ["#800000", "#191970", "#006400", "#483D8B", "#8B0000"]},
    "olive": {"description": "Warm metallics and soft tones — copper, gold, and rose — bring balance.",
              "colors": ["#B87333", "#FFD700", "#E6BE8A", "#D4AF37", "#DAA520"]},
    "tan": {"description": "Tropical and bright shades like teal, coral, and white are perfect for you.",
            "colors": ["#20B2AA", "#FF7F50", "#F5F5F5", "#FFA07A", "#40E0D0"]},
    "deep": {"description": "Royal shades — blue, emerald, magenta — complement deep tones best.",
             "colors": ["#00008B", "#228B22", "#8B008B", "#800080", "#9932CC"]},
    "dark": {"description": "Bold contrasts like white, gold, and red make your tone pop beautifully.",
             "colors": ["#FFFFFF", "#FFD700", "#DC143C", "#FF8C00", "#32CD32"]},
    "very_dark": {"description": "Vibrant and metallic tones — cobalt, fuchsia, bronze — look stunning.",
                  "colors": ["#0033A0", "#FF00FF", "#CD7F32", "#FFD700", "#800000"]}
}

# ----------------------------
# Streamlit App UI
# ----------------------------
st.title("🎨 Skin Tone Detection & Color Recommendation")
st.write("Upload an image — the app will automatically detect your face, classify your skin tone, and show matching outfit colors with a saliency map!")

uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')

    with st.spinner("🔍 Detecting face..."):
        cropped_face, detected_img = detect_and_crop_face(image)

    if cropped_face is None:
        st.error("❌ No face detected. Please upload a clear image showing your face.")
    else:
        # Predict instantly (no button)
        with st.spinner("Analyzing skin tone... please wait..."):
            preds, img_tensor = predict_img(cropped_face)
            pred_class = np.argmax(preds)
            confidence = np.max(preds) * 100
            saliency = get_saliency_map(model, img_tensor)

        class_labels = [
            "very_light", "light", "light_medium", "medium", "medium_deep",
            "olive", "tan", "deep", "dark", "very_dark"
        ]
        label = class_labels[pred_class]

        # Show all images
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(image, caption="🖼️ Original Image", width=250)
        with col2:
            st.image(cropped_face, caption="✂️ Cropped Face", width=250)
        with col3:
            st.image(saliency, caption="🔥 Saliency Map", width=250, clamp=True)

        # Results
        st.markdown(f"<h2 style='color:#4CAF50;'>🏷️ Predicted Skin Tone</h2>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align:center; color:#FF5722;'>{label.replace('_', ' ').title()}</h1>", unsafe_allow_html=True)
        st.write(f"**Confidence:** {confidence:.2f}%")
        st.progress(int(confidence))

        # Recommendations
        recommendation = color_recommendations[label]
        st.markdown(f"<h3 style='margin-top:25px;'>🎨 Recommended Colors</h3>", unsafe_allow_html=True)
        st.write(recommendation["description"])

        color_cols = st.columns(len(recommendation["colors"]))
        for i, color in enumerate(recommendation["colors"]):
            with color_cols[i]:
                st.markdown(
                    f"<div style='background-color:{color}; width:100%; height:60px; border-radius:10px; border:1px solid #ccc;'></div>",
                    unsafe_allow_html=True
                )

# ----------------------------
# Footer
# ----------------------------
st.markdown("""
---
✅ **Built with Streamlit** | 💻 **Model:** DenseNet (Skin Tone Classifier) | 📸 **Author:** Muhammad Hassan
""")
