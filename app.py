
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

st.set_page_config(page_title="Chat ou Chien ?", page_icon="🐾")
st.title("🐱 Chat ou Chien 🐶")
st.write("Chargez une image et le modele predit la classe.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model_cats_dogs.keras")

model = load_model()
IMG_SHAPE = (150, 150)

file = st.file_uploader("Image (jpg/png)", type=["jpg", "jpeg", "png"])
if file is not None:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Image chargee", use_column_width=True)

    arr = np.array(img.resize(IMG_SHAPE))
    arr = preprocess_input(arr.astype("float32"))
    arr = np.expand_dims(arr, axis=0)

    proba = float(model.predict(arr)[0][0])
    classe = "CHIEN 🐶" if proba > 0.5 else "CHAT 🐱"
    confiance = proba if proba > 0.5 else 1 - proba

    st.subheader(f"Prediction : {classe}")
    st.write(f"Confiance : {confiance*100:.1f}%")
    st.progress(confiance)
