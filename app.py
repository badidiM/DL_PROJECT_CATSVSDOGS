import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense

st.set_page_config(page_title="Chat ou Chien ?", page_icon="🐾")
st.title("🐱 Chat ou Chien 🐶")
st.write("Chargez une image et le modele predit la classe.")

IMG_SHAPE = (150, 150)
INPUT_SHAPE = (150, 150, 3)

@st.cache_resource
def load_model():
    # Reconstruire EXACTEMENT la meme architecture qu'a l'entrainement
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=INPUT_SHAPE)
    base_model.trainable = False
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.3),
        Dense(128, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.load_weights("model_cats_dogs.weights.h5", skip_mismatch=True)
    return model

model = load_model()

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