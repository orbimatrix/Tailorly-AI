import cv2
import mediapipe as mp
import math
import numpy as np
import base64

class BodyMeasurement:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

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
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
        left_heel = landmarks[self.mp_pose.PoseLandmark.LEFT_HEEL]
        right_heel = landmarks[self.mp_pose.PoseLandmark.RIGHT_HEEL]
        
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]

        # Calculate Pixel Height (approximate)
        y_coords = [lm.y for lm in landmarks]
        min_y = min(y_coords)
        max_y = max(y_coords)
        
        pixel_height = (max_y - min_y) * image_height
        
        if pixel_height == 0:
             return {"error": "Invalid pixel height calculation"}

        # Scale Factor: cm per pixel
        scale_factor = user_height_cm / pixel_height

        # Helper to get pixel coordinates
        def to_pixel(lm):
            return (int(lm.x * image_width), int(lm.y * image_height))

        # Calculate Widths (Horizontal distance)
        def get_dist_cm(p1, p2):
            dist_px = math.sqrt(((p1.x - p2.x) * image_width)**2 + ((p1.y - p2.y) * image_height)**2)
            return dist_px * scale_factor

        shoulder_width = get_dist_cm(left_shoulder, right_shoulder)
        hip_width = get_dist_cm(left_hip, right_hip)
        
        # Arm Length (Shoulder to Wrist)
        left_arm = get_dist_cm(left_shoulder, left_wrist)
        right_arm = get_dist_cm(right_shoulder, right_wrist)
        avg_arm = (left_arm + right_arm) / 2

        # --- VISUALIZATION ---
        # Draw landmarks
        self.mp_drawing.draw_landmarks(
            image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        # Draw measurement lines
        def draw_measurement(p1, p2, value, label):
            pt1 = to_pixel(p1)
            pt2 = to_pixel(p2)
            cv2.line(image, pt1, pt2, (0, 255, 0), 2)
            mid_point = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
            cv2.putText(image, f"{label}: {value}cm", mid_point, 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        draw_measurement(left_shoulder, right_shoulder, round(shoulder_width, 1), "Shoulder")
        draw_measurement(left_hip, right_hip, round(hip_width, 1), "Hip")
        draw_measurement(left_shoulder, left_wrist, round(left_arm, 1), "L. Arm")
        
        # Encode image to base64
        _, buffer = cv2.imencode('.jpg', image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return {
            "height_cm": user_height_cm,
            "shoulder_width_cm": round(shoulder_width, 2),
            "hip_width_cm": round(hip_width, 2),
            "arm_length_cm": round(avg_arm, 2),
            "scale_factor": scale_factor,
            "image_base64": img_base64
        }
