import streamlit as st
import cv2
import numpy as np
import pyttsx3
from ultralytics import YOLO
from PIL import Image
from pathlib import Path
import time

# 🎨 Set Streamlit page config
st.set_page_config(page_title="Real-Time Object Detection", layout="wide", page_icon="📸")

# -----------------------------
# 🧑 Sidebar with Project Info (direct) and About Me (expandable)
# -----------------------------
with st.sidebar:
    st.image("abhishek.jpeg", width=200)

    st.markdown("""
This app uses **YOLOv8** to detect objects in real time from your webcam and gives **voice announcements** every few seconds.

### 🔧 Features:
- 🚀 YOLOv8 real-time object detection
- 🗣️ Voice alerts for detected objects
- 📷 Camera control buttons (Start/Stop)
- 🧠 Smart cooldown prevents repeat messages
    """)

    with st.expander("👤 About Me"):
        st.markdown("""
**Abhishek Wekhande**  
Final Year BTech Student  
Symbiosis Institute of Technology  
📧 [abhishek.wekhande20@gmail.com](mailto:abhishek.wekhande20@gmail.com)  
📞 +91-9876543210  
🔗 [GitHub](https://github.com/abhishek-wekhande)
        """)

# -----------------------------
# 📸 Main Title
# -----------------------------
st.markdown("<h1 style='text-align: center;'>📸 Real-Time Object Detection with Voice</h1>", unsafe_allow_html=True)

# -----------------------------
# 📦 Load YOLOv8 model
# -----------------------------
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

# -----------------------------
# 🗣️ Voice Engine
# -----------------------------
engine = pyttsx3.init()
engine.setProperty('rate', 150)
last_spoken = {}

# -----------------------------
# ▶️ Camera Controls
# -----------------------------
col1, col2 = st.columns([1, 1])
with col1:
    start = st.button("▶️ Start Camera")
with col2:
    stop = st.button("⏹️ Stop Camera")

FRAME_WINDOW = st.empty()

if "camera_on" not in st.session_state:
    st.session_state.camera_on = False

if start:
    st.session_state.camera_on = True
if stop:
    st.session_state.camera_on = False

# -----------------------------
# 🔁 Camera + Detection Loop
# -----------------------------
if st.session_state.camera_on:
    cap = cv2.VideoCapture(0)
    detected_labels = set()

    while st.session_state.camera_on:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Could not access the webcam.")
            break

        results = model(frame)[0]
        annotated_frame = results.plot()

        labels_in_frame = set()
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            labels_in_frame.add(label)

        for label in labels_in_frame:
            now = time.time()
            if label not in last_spoken or now - last_spoken[label] > 5:
                try:
                    engine.stop()
                    engine.say(f"{label} detected")
                    engine.runAndWait()
                    last_spoken[label] = now
                except RuntimeError:
                    pass

        detected_labels = labels_in_frame
        FRAME_WINDOW.image(annotated_frame, channels="BGR")

    cap.release()
else:
    st.info("Click ▶️ Start Camera to begin.")

# -----------------------------
# ❤️ Footer
# -----------------------------
st.markdown("""
<hr style="border: none; border-top: 1px solid #bbb;">
<div style='text-align: center; font-size: 16px;'>
    Made with ❤️ by <b>Abhishek</b>
</div>
""", unsafe_allow_html=True)
