import streamlit as st
import cv2
import numpy as np
import pyttsx3
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="Real-Time Object Detection", layout="centered")
st.title("📸 Real-Time Object Detection with Voice")

# Load model
@st.cache_resource
def load_model():
    return YOLO("model.pt")  # Make sure model.pt is in the same folder

model = load_model()

# Initialize voice engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Start webcam
run = st.checkbox("Start Camera")

FRAME_WINDOW = st.image([])

if run:
    cap = cv2.VideoCapture(0)
    detected_labels = set()

    while run:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        annotated_frame = results.plot()

        labels_in_frame = set()
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            labels_in_frame.add(label)

        # Speak only new labels
        new_labels = labels_in_frame - detected_labels
        for label in new_labels:
            engine.say(f"{label} detected")
            engine.runAndWait()
        detected_labels = labels_in_frame

        FRAME_WINDOW.image(annotated_frame, channels="BGR")

    cap.release()
else:
    st.write("Click the checkbox to start camera.")

