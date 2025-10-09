import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import cv2

# ----------------------------
# ✅ Must be the first Streamlit command
# ----------------------------
st.set_page_config(page_title="Image Classifier", page_icon="🖼️", layout="centered")

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_trained_model():
    model = load_model("densenet_model.keras")
    return model

model = load_trained_model()

# ----------------------------
# Helper Function for Prediction
# ----------------------------
def predict_img(image):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    preds = model.predict(img)
    return preds

# ----------------------------
# Streamlit App UI
# ----------------------------
st.title("🧠 Image Classification App")
st.write("Upload an image below and the model will automatically classify it.")

uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')

    # Two-column layout
    col1, col2 = st.columns([1, 1])

    # Left column → Uploaded image
    with col1:
        st.image(image, caption="Uploaded Image", width=280, use_container_width=False)

    # Right column → Prediction results
    with col2:
        with st.spinner("Analyzing image... please wait..."):
            preds = predict_img(image)
            pred_class = np.argmax(preds)
            confidence = np.max(preds) * 100

        # Realistic 10-class labels
        class_labels = [
            "very_light",   #1
            "light",        #2
            "light_medium", #3
            "medium",       #4
            "medium_deep",  #5
            "olive",        #6
            "tan",          #7
            "deep",         #8
            "dark",         #9
            "very_dark"     #10
        ]

        label = class_labels[pred_class]

        # Display prediction
        st.markdown(f"<h2 style='color:#4CAF50;'>🏷️ Predicted Class:</h2>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align:center; color:#FF5722;'>{label.replace('_', ' ').title()}</h1>", unsafe_allow_html=True)
        st.write(f"**Confidence:** {confidence:.2f}%")
        st.progress(int(confidence))

# ----------------------------
# Footer
# ----------------------------
st.markdown("""
---
✅ **Built with Streamlit** | 💻 **Model:** DenseNet | 📸 **Author:** Muhammad Hassan
""")
