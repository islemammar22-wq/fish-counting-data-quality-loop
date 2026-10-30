"""Streamlit demo: upload an image, get the object count with boxes drawn.

Run: streamlit run app.py
"""
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.title("Object counter")

weights = st.text_input("Weights path", "runs/round2/weights/best.pt")
conf = st.slider("Confidence threshold", 0.05, 0.9, 0.25, 0.05)
upload = st.file_uploader("Image", type=["jpg", "jpeg", "png"])


@st.cache_resource
def load(path):
    return YOLO(path)


if upload is not None:
    image = Image.open(upload).convert("RGB")
    try:
        model = load(weights)
    except Exception as e:
        st.error(f"Could not load weights: {e}")
        st.stop()
    result = model.predict(np.array(image), conf=conf, verbose=False)[0]
    annotated = result.plot()[:, :, ::-1]  # BGR -> RGB
    st.image(annotated, caption=f"Count: {len(result.boxes)}")
    st.metric("Count", len(result.boxes))
