import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from pathlib import Path
import platform

st.set_page_config(page_title="Real-Time Object Detection", layout="centered")
st.title("📸 Real-Time Object Detection with Voice")

# ✅ Conditionally import voice engine only for local use
if platform.system() != "Linux" or "streamlit" not in st.runtime.scriptrunner.get_script_run_ctx().runtime:
    import pyttsx3
    voice_enabled = True
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
else:
    st.warning("Voice output is disabled on Streamlit Cloud.")
    voice_enabled = False

# ✅ Load or download YOLO model
@st.cache_resource
def load_model():
    model_path = Path("yolov8n.pt")
    if not model_path.exists():
        with st.spinner("Downloading YOLOv8 model..."):
            import urllib.request
            url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
            urllib.request.urlretrieve(url, model_path)
    return YOLO(str(model_path))

model = load_model()

# ✅ Start webcam
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

        # ✅ Speak new labels (only locally)
        new_labels = labels_in_frame - detected_labels
        if voice_enabled:
            for label in new_labels:
                engine.say(f"{label} detected")
                engine.runAndWait()

        detected_labels = labels_in_frame
        FRAME_WINDOW.image(annotated_frame, channels="BGR")

    cap.release()
else:
    st.write("Click the checkbox to start camera.")
