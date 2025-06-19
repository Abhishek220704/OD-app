import streamlit as st
import os
import gdown  # ✅ Google Drive downloader

# Google Drive direct download URL (public file)
weights_url = "https://drive.google.com/uc?id=1MPzq3LQ7Hy2Nkv7p9akIFZi7tvkKZdbU"

# Check and download yolov3.weights
if not os.path.exists("yolov3.weights"):
    with st.spinner("Downloading yolov3.weights from Google Drive..."):
        try:
            import gdown
        except ImportError:
            st.error("Please add 'gdown' to your requirements.txt")
            raise

        gdown.download(weights_url, "yolov3.weights", quiet=False)
        st.success("✅ Download complete!")


import cv2
import numpy as np
import pyttsx3
from PIL import Image

# Initialize TTS engine
engine = pyttsx3.init()

# Load YOLOv3 model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Streamlit UI
st.title("🧠 Real-Time Object Detection with YOLOv3 + Voice")
run = st.checkbox('Start Camera')
FRAME_WINDOW = st.image([])

# Start webcam
cap = cv2.VideoCapture(0)
spoken_labels = set()

while run:
    ret, frame = cap.read()
    if not ret:
        st.warning("❌ Unable to access webcam")
        break

    height, width, _ = frame.shape

    # Convert to blob for YOLO
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        color = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{label} {int(confidence*100)}%", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Voice output once per label
        if label not in spoken_labels:
            spoken_labels.add(label)
            engine.say(f"{label} detected")
            engine.runAndWait()

    FRAME_WINDOW.image(frame, channels="BGR")

else:
    cap.release()
    st.write("⛔ Camera stopped.")
