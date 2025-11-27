import cv2
import mediapipe as mp
import math
import numpy as np

class BodyMeasurement:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

    def calculate_distance(self, p1, p2):
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def process_image(self, image_bytes, user_height_cm):
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Could not decode image")

        image_height, image_width, _ = image.shape
        
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.pose.process(image_rgb)
        
        if not results.pose_landmarks:
            return {"error": "No body detected"}

        landmarks = results.pose_landmarks.landmark

        # Get key landmarks
        # MediaPipe Pose Landmarks: https://developers.google.com/mediapipe/solutions/vision/pose
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
        left_heel = landmarks[self.mp_pose.PoseLandmark.LEFT_HEEL]
        right_heel = landmarks[self.mp_pose.PoseLandmark.RIGHT_HEEL]
        
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]

        # Calculate Pixel Height (approximate from nose to average heel position)
        # Note: This is a rough approximation. A full body bounding box would be better, 
        # but using landmarks is a good start.
        # We'll use the Y-coordinate difference.
        
        avg_heel_y = (left_heel.y + right_heel.y) / 2
        # Top of head is roughly above the nose. 
        # Let's assume the distance from nose to top of head is roughly 1/7th of the head-to-chin distance?
        # For simplicity, let's use the full vertical span of landmarks or just Nose to Heel and add a small buffer.
        # A better heuristic: Eye to Heel is often used, or just bounding box height.
        
        # Let's use the bounding box of the detected pose for "height" in pixels
        y_coords = [lm.y for lm in landmarks]
        min_y = min(y_coords)
        max_y = max(y_coords)
        
        pixel_height = (max_y - min_y) * image_height
        
        if pixel_height == 0:
             return {"error": "Invalid pixel height calculation"}

        # Scale Factor: cm per pixel
        scale_factor = user_height_cm / pixel_height

        # Calculate Widths (Horizontal distance)
        # We use x coordinates * image_width
        
        def get_dist_cm(p1, p2):
            # Euclidean distance in pixels
            dist_px = math.sqrt(((p1.x - p2.x) * image_width)**2 + ((p1.y - p2.y) * image_height)**2)
            return dist_px * scale_factor

        shoulder_width = get_dist_cm(left_shoulder, right_shoulder)
        hip_width = get_dist_cm(left_hip, right_hip)
        
        # Arm Length (Shoulder to Wrist)
        left_arm = get_dist_cm(left_shoulder, left_wrist)
        right_arm = get_dist_cm(right_shoulder, right_wrist)
        avg_arm = (left_arm + right_arm) / 2

        return {
            "height_cm": user_height_cm,
            "shoulder_width_cm": round(shoulder_width, 2),
            "hip_width_cm": round(hip_width, 2),
            "arm_length_cm": round(avg_arm, 2),
            "scale_factor": scale_factor
        }
