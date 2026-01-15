import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import cv2
import os

# -----------------------------------------------------------
# ✅ Streamlit Setup
# -----------------------------------------------------------
st.set_page_config(page_title="Skin Tone Detector", page_icon="🎨", layout="centered")

st.title("🎨 Skin Tone Detection & Outfit Color Suggestions")
st.write("Capture or upload your image to detect your **skin tone** and get **personalized outfit color recommendations**.")

# -----------------------------------------------------------
# ✅ Load Trained Model
# -----------------------------------------------------------
@st.cache_resource
def load_trained_model():
    model = load_model("skin_tone_cnn_model.keras")  
    return model

model = load_trained_model()

# -----------------------------------------------------------
# ✅ Load Haar Cascade for Face Detection
# -----------------------------------------------------------
cascade_path = os.path.join(os.getcwd(), "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    st.error("❌ Haar Cascade not found. Check haarcascade_frontalface_default.xml path.")
    st.stop()

# -----------------------------------------------------------
# ✅ Face Detection Helper
# -----------------------------------------------------------
def detect_crop_face(image_pil):
    img = np.array(image_pil)
    img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    if len(faces) == 0:
        return None, image_pil

    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    face = img_cv[y:y+h, x:x+w]
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    # draw rectangle
    cv2.rectangle(img_cv, (x, y), (x+w, y+h), (0, 255, 0), 3)
    detected_img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

    return Image.fromarray(face_rgb), detected_img

# -----------------------------------------------------------
# ✅ Prediction Helper
# -----------------------------------------------------------
def predict_skin_tone(face_crop):
    img = np.array(face_crop)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img_tensor = np.expand_dims(img, axis=0)
    preds = model.predict(img_tensor)
    return preds
# -----------------------------------------------------------
# 🎨 Color Recommendations (Expanded – Color Wheel Based)
# -----------------------------------------------------------
color_recommendations = {
    "Light": {
        "description": (
            "Light skin tones look best in soft, low-contrast shades. "
            "Pastels, cool hues, and light neutrals maintain balance without overpowering."
        ),
        "colors": [
            "#D8BFD8",  # Thistle
            "#B0E0E6",  # Powder Blue
            "#ADD8E6",  # Light Blue
            "#924B05",  # Brown
            "#FADADD",  # Soft Rose
            "#C1E1C1",  # Mint Green
            "#E0FFFF",  # Light Cyan
            "#F5F5DC",  # Beige (neutral)
            "#FAF0E6"   # Linen
        ]
    },

    "Fair": {
        "description": (
            "Fair skin tones benefit from warm pastels and soft jewel tones. "
            "These hues add warmth while keeping contrast gentle and flattering."
        ),
        "colors": [
            "#834814",  # Sandy Brown
            "#F4C2C2",  # Rose Pink
            "#E6B8A2",  # Nude Peach
            "#66CDAA",  # Medium Aquamarine
            "#D8BFD8",  # Soft Purple
            "#9370DB",  # Medium Purple
            "#FFF0DC",  # Cream (neutral)
            "#D2B48C"   # Tan
        ]
    },

    "Medium": {
        "description": (
            "Medium skin tones shine with warm, earthy shades and rich colors. "
            "Analogous and split-complementary tones enhance depth and glow."
        ),
        "colors": [
            "#808000",  # Olive Green
            "#556B2F",  # Dark Olive
            "#D2B48C",  # Tan
            "#DAA520",  # Goldenrod
            "#B8860B",  # Dark Gold
            "#A0522D",  # Sienna
            "#008080",  # Teal (contrast pop)
            "#4682B4"   # Steel Blue
        ]
    },

    "Dark": {
        "description": (
            "Dark skin tones are enhanced by bold, saturated, and high-contrast colors. "
            "Jewel tones and bright accents create striking visual harmony."
        ),
        "colors": [
            "#FFD700",  # Gold
            "#FF8C00",  # Dark Orange
            "#FFFFFF",  # White (high contrast)
            "#D2B48C",  # Tan
            "#8B0000",  # Deep Red
            "#006400",  # Dark Green
            "#00CED1",  # Dark Turquoise
            "#1E90FF"   # Dodger Blue
        ]
    }
}


# -----------------------------------------------------------
# 📸 Choose Image Source
# -----------------------------------------------------------
option = st.radio("Choose Image Source:", ["📷 Camera", "🖼️ Gallery"])

image = None
face_crop = None
detected_img = None

if option == "📷 Camera":
    camera_image = st.camera_input("Take a picture")
    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")
        face_crop, detected_img = detect_crop_face(image)

elif option == "🖼️ Gallery":
    uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        face_crop, detected_img = detect_crop_face(image)

# -----------------------------------------------------------
# 🎯 Process Prediction
# -----------------------------------------------------------
if image is not None:
    if face_crop is None:
        st.error("❌ No face detected. Please try again with a clearer image.")
    else:
        st.success("✅ Face detected successfully!")

        col1, col2 = st.columns(2)
        with col1:
            st.image(detected_img, caption="Detected Face", use_column_width=True)
        with col2:
            st.image(face_crop, caption="Cropped Face", use_column_width=True)

        # Prediction
        with st.spinner("🎯 Analyzing skin tone..."):
            preds = predict_skin_tone(face_crop)
            pred_class = np.argmax(preds)
            confidence = np.max(preds) * 100

        class_labels = ["Light", "Fair", "Medium", "Dark"]
        label = class_labels[pred_class]

        st.markdown(f"## 🏷️ Predicted Skin Tone: **{label.upper()}**")
        st.write(f"**Confidence:** {confidence:.2f}%")
        st.progress(int(confidence))

        recommendation = color_recommendations[label]
        st.markdown("### 🎨 Recommended Outfit Colors")
        st.write(recommendation["description"])

        color_cols = st.columns(len(recommendation["colors"]))
        for i, color in enumerate(recommendation["colors"]):
            with color_cols[i]:
                st.markdown(
                    f"<div style='background-color:{color}; width:100%; height:60px; border-radius:10px;'></div>",
                    unsafe_allow_html=True
                )

st.markdown("""
---
👨‍💻 **Developed by Logical Technologist**  
""")
