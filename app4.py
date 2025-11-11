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
    model = load_model("densenet_skin_tone_final_model.keras")  
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
# 🎨 Color Recommendations
# -----------------------------------------------------------
color_recommendations = {
    "Light": {
        "description": "Soft pastels and airy colors like lavender, sky blue, and peach enhance your tone.",
        "colors": ["#E6E6FA", "#FFDAB9", "#B0E0E6", "#F5DEB3", "#FFB6C1"]
    },
    "Fair": {
        "description": "Warm and subtle tones — coral, rose, and turquoise work beautifully.",
        "colors": ["#FF7F50", "#AFEEEE", "#E0B0FF", "#FFDAB9", "#F5DEB3"]
    },  
    "Medium": {
        "description": "Rich earthy shades like rust, olive, and caramel bring out your warmth.",
        "colors": ["#B7410E", "#808000", "#DAA520", "#8B4513", "#CD853F"]
    },
    "Dark": {
        "description": "Bold contrasts — white, gold, and deep red make your tone pop beautifully.",
        "colors": ["#FFFFFF", "#FFD700", "#8B0000", "#FF8C00", "#228B22"]
    },
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
