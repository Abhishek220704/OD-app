# 📸 Real-Time Object Detection with Voice Output

This project is a **real-time object detection system** using **YOLOv8** and **Streamlit**, capable of recognizing objects from your webcam feed and speaking the names of detected objects aloud.

Developed by **Abhishek Wekhande** – Final Year BTech Student, Symbiosis Institute of Technology.

---

## 🧠 Project Overview

This application utilizes:
- **YOLOv8**: State-of-the-art real-time object detection model
- **OpenCV**: For capturing webcam input
- **Streamlit**: For web-based UI and deployment
- **pyttsx3**: For voice output (text-to-speech)

---

## 🚀 Features

- 📷 Live webcam feed processing
- 🧠 Smart label tracking
- 🗣️ Voice alerts every 5 seconds for persistent object detection
- 🧭 Sidebar with information and controls
- ⚙️ Start/Stop camera functionality via UI

---

## 🖼️ Sample Results

> Below are some real-time detection screenshots generated using this application:

![Sample 1](assets/ss1.jpg)  
![Sample 2](assets/ss3.jpg)
![Sample 2](assets/ss4.jpg)

Make sure these files (`sample1.jpg`, `sample2.jpg`) are placed inside a folder named `assets/` in your project directory.

---

## 💡 Requirements

Create a `requirements.txt` with:

```txt
streamlit
opencv-python
ultralytics
numpy
pyttsx3
Pillow
```

## To install:  
```
pip install -r requirements.txt
```

How to run the application locally:  
1. Ensure `app.py`, `yolov8n.pt`, and `abhishek.jpeg` are in the same folder.  
2. Run:
```streamlit run app.py```  
4. Make sure your webcam is accessible and not used by another app.

**Conditions for Best Performance**:  
Run the application in a well-lit environment. Detection accuracy may reduce in low light or motion blur. Python 3.8+ is recommended.

Folder structure:
```
real-time-object-detection/
├── app.py
├── requirements.txt
├── yolov8n.pt
├── abhishek.jpeg
├── README.md
└── assets/
├── sample1.jpg
└── sample2.jpg
```

---

**Made with ❤️ by Abhishek Wekhande**  
📧 abhishek.wekhande20@gmail.com  
🔗 [GitHub](https://github.com/abhishek-wekhande)
