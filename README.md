# Realtime-FHP-Detector

> **A computer vision-based system for real-time Forward Head Posture (FHP) detection.**
> This project monitors user posture through a webcam and analyzes the cervical angle to detect FHP (commonly known as "Text Neck").

---

## Visualization

Here is how the system processes the visual data using landmark detection.

| Holistic Landmark Detection | Specific Keypoint Extraction |
| :---: | :---: |
| <img src="https://github.com/user-attachments/assets/e8b8f1dc-dcce-479a-9598-84c001f30686" width="500"> | <img src="https://github.com/user-attachments/assets/2d4f0bd8-fa7b-41ff-9dd8-ac6c4bd56bf8" width="500"> |
| *Real-time face & pose landmark tracking* | *Extracting specific coordinates for angle calculation* |

---

## Key Features
* **Real-time Tracking**: High-performance detection using Google MediaPipe.
* **Visual Overlay**: Real-time visualization of skeleton landmarks and connection lines for immediate feedback.
* **Accuracy-driven**: Optimized to detect subtle shifts in head position (e.g., detecting a 12° forward tilt).

## 🛠 Tech Stack
* **Language**: Python 3.10
* **Core Libraries**: `OpenCV`, `MediaPipe`, `NumPy`
* **Development Environment**: Conda

## Getting Started

### Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/hiksh/Realtime-FHP-Detector.git](https://github.com/hiksh/Realtime-FHP-Detector.git)
   cd Realtime-FHP-Detector

2. Install the required dependencies using the requirements.txt file:
   ```bash
   pip install -r requirements.txt

### Execution
Run the main detection script:
   ```bash
   python main.py
