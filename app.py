import os
os.environ["KERAS_BACKEND"] = "jax"

import streamlit as st
import numpy as np
from PIL import Image
import h5py

# ==============================
# CONFIGURATION
# ==============================

MODEL_PATH = "mobilenetv2_fruits.h5"

CLASS_NAMES = ["Apple", "Banana", "Mango", "Orange", "Strawberry"]

CLASS_EMOJI = {
    "Apple":      "\U0001F34E",
    "Banana":     "\U0001F34C",
    "Mango":      "\U0001F96D",
    "Orange":     "\U0001F34A",
    "Strawberry": "\U0001F353"
}

IMAGE_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 70.0
MARGIN_THRESHOLD = 30.0

# ==============================
# LOAD MODEL — use custom_objects to fix DepthwiseConv2D JAX issue
# ==============================

@st.cache_resource
def load_model():
    import keras
    from keras.src.layers import DepthwiseConv2D as BaseDepthwiseConv2D

    # Patch: wrap load to ignore 'groups' arg that JAX backend can't handle
    class PatchedDepthwiseConv2D(BaseDepthwiseConv2D):
        def __init__(self, *args, **kwargs):
            kwargs.pop("groups", None)
            super().__init__(*args, **kwargs)

    model = keras.models.load_model(
        MODEL_PATH,
        custom_objects={"DepthwiseConv2D": PatchedDepthwiseConv2D},
        compile=False
    )
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
    st.image(image, caption="Uploaded Image", width="stretch")

    img_resized = image.resize(IMAGE_SIZE)
    img_array = np.array(img_resized, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("\U0001F50D Analyzing image..."):
        predictions = model(img_array, training=False)
        probs = np.array(predictions[0])

    predicted_index = int(np.argmax(probs))
    predicted_class = CLASS_NAMES[predicted_index]
    top_confidence = float(probs[predicted_index]) * 100

    sorted_probs = np.sort(probs)[::-1]
    second_confidence = float(sorted_probs[1]) * 100
    margin = top_confidence - second_confidence

    is_fruit = (top_confidence >= CONFIDENCE_THRESHOLD) and (margin >= MARGIN_THRESHOLD)

    st.divider()
    st.subheader("\U0001F3AF Prediction Result")

    if is_fruit:
        emoji = CLASS_EMOJI[predicted_class]
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"{emoji} **{predicted_class}**")
            st.caption("Predicted Class")
        with col2:
            st.info(f"**{top_confidence:.2f}%**")
            st.caption("Confidence Score")

        st.divider()
        st.subheader("\U0001F4CA All Class Probabilities")
        for i, cls in enumerate(CLASS_NAMES):
            prob = float(probs[i]) * 100
            e = CLASS_EMOJI[cls]
            st.progress(int(prob), text=f"{e} {cls}: {prob:.2f}%")

    else:
        col1, col2 = st.columns(2)
        with col1:
            st.error("\U0001F6AB **Not a Fruit**")
            st.caption("Predicted Class")
        with col2:
            st.warning(f"**{top_confidence:.2f}%**")
            st.caption("Low Confidence Score")
        st.info(
            "\U0001F4A1 The uploaded image does not appear to be one of the 5 recognized fruits "
            "(Apple, Banana, Mango, Orange, Strawberry). Please upload a clear fruit photo."
        )

else:
    st.info("\U0001F446 Please upload a fruit image to get started.")

st.divider()
st.caption("Lab Activity 7 | Model: MobileNetV2 | Dataset: Fruits (150 images | 5 classes)")
