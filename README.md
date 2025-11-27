# TailorAI - AI-Powered Body Measurement

TailorAI is a web application that leverages computer vision and machine learning to estimate body measurements from a single photo. It is designed to help users get accurate sizing for clothing without manual measuring tapes.

## Features

- **Instant Body Measurement**: Estimates shoulder width, hip width, and arm length using a webcam or uploaded photo.
- **AI-Powered**: Uses Google's MediaPipe Pose for state-of-the-art landmark detection.
- **Privacy-First**: Images are processed in memory and not permanently stored.
- **User-Friendly Interface**: Simple web interface with real-time camera preview.

## Tech Stack

- **Backend**: Python, FastAPI, OpenCV, MediaPipe
- **Frontend**: HTML5, CSS3, JavaScript
- **Deployment**: Uvicorn (ASGI Server)

## Installation

1. **Clone the repository** (or download the files):
   ```bash
   git clone <repository-url>
   cd tailorAI
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.8+ installed.
   ```bash
   pip install mediapipe opencv-python fastapi uvicorn python-multipart
   ```

## Usage

1. **Start the Server**:
   ```bash
   python app.py
   ```

2. **Access the App**:
   Open your browser and navigate to: `http://localhost:8000`

3. **Get Measured**:
   - Enter your height in centimeters (e.g., 175).
   - Allow camera access.
   - Stand back until your full body is visible.
   - Click **Capture & Measure**.

## How It Works

1. The app captures an image from your webcam.
2. It asks for your height to establish a scale (pixels to cm).
3. MediaPipe Pose detects 33 key body landmarks.
4. The system calculates distances between specific landmarks (e.g., shoulders) and converts them to centimeters using your height as a reference.

## Limitations

- **Accuracy**: Measurements are estimates. Accuracy depends on camera angle, distance, and pose.
- **Scaling**: Currently relies on user-provided height. For higher precision, a reference object (like a credit card) in the frame would be required.
