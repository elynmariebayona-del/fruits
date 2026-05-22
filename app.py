import os
os.environ["KERAS_BACKEND"] = "jax"

import streamlit as st
import keras
import numpy as np
from PIL import Image

# ==============================
# CONFIGURATION
# ==============================

MODEL_PATH = "mobilenetv2_fruits.h5"

CLASS_NAMES = [
    "Apple",
    "Banana",
    "Mango",
    "Orange",
    "Strawberry"
]

CLASS_EMOJI = {
    "Apple":      "\U0001F34E",
    "Banana":     "\U0001F34C",
    "Mango":      "\U0001F96D",
    "Orange":     "\U0001F34A",
    "Strawberry": "\U0001F353"
}

IMAGE_SIZE = (224, 224)

# If the top prediction is below this threshold, show "Not a fruit"
CONFIDENCE_THRESHOLD = 60.0  # 60%

# ==============================
# LOAD MODEL
# ==============================

@st.cache_resource
def load_model():
    model = keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="Fruit Classifier",
    page_icon="\U0001F34E",
    layout="centered"
)

# ==============================
# HEADER
# ==============================

st.title("\U0001F34E Fruit Image Classifier")
st.markdown("### Lab Activity 7 — Transfer Learning with MobileNetV2")
st.markdown("Upload a photo of a fruit and the model will identify it.")
st.markdown("**Classes:** Apple \U0001F34E | Banana \U0001F34C | Mango \U0001F96D | Orange \U0001F34A | Strawberry \U0001F353")
st.divider()

# ==============================
# FILE UPLOAD
# ==============================

uploaded_file = st.file_uploader(
    "\U0001F4E4 Upload a fruit image (JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_resized = image.resize(IMAGE_SIZE)
    img_array = np.array(img_resized, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)

    with st.spinner("\U0001F50D Analyzing image..."):
        predictions = model.predict(img_array)

    predicted_index = int(np.argmax(predictions[0]))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(predictions[0][predicted_index]) * 100

    st.divider()
    st.subheader("\U0001F3AF Prediction Result")

    # ==============================
    # CONFIDENCE CHECK
    # ==============================

    if confidence < CONFIDENCE_THRESHOLD:
        # Not confident enough — not a recognized fruit
        col1, col2 = st.columns(2)
        with col1:
            st.error("\U0001F6AB **Not a Fruit**")
            st.caption("Predicted Class")
        with col2:
            st.warning(f"**{confidence:.2f}%**")
            st.caption("Confidence Score (too low)")

        st.info(
            "\U0001F4A1 The model could not recognize this as one of the 5 fruit classes "
            "(Apple, Banana, Mango, Orange, Strawberry). "
            "Please upload a clearer fruit image."
        )

    else:
        # Confident — show result normally
        emoji = CLASS_EMOJI[predicted_class]

        col1, col2 = st.columns(2)
        with col1:
            st.success(f"{emoji} **{predicted_class}**")
            st.caption("Predicted Class")
        with col2:
            st.info(f"**{confidence:.2f}%**")
            st.caption("Confidence Score")

    # Always show all class probabilities
    st.divider()
    st.subheader("\U0001F4CA All Class Probabilities")

    for i, cls in enumerate(CLASS_NAMES):
        prob = float(predictions[0][i]) * 100
        e = CLASS_EMOJI[cls]
        st.progress(
            int(prob),
            text=f"{e} {cls}: {prob:.2f}%"
        )

else:
    st.info("\U0001F446 Please upload a fruit image to get started.")

st.divider()
st.caption("Lab Activity 7 | Model: MobileNetV2 | Dataset: Fruits (150 images | 5 classes)")
