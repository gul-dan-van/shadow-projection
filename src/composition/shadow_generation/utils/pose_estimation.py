# pose_estimation.py

import cv2
import numpy as np
import mediapipe as mp


mp_pose = mp.solutions.pose
POSE = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def get_feet_coords(image, pose=POSE, pose_indices=[29, 30, 31, 32]):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    height, width, _ = image.shape

    if not results.pose_landmarks:
        # Instead of raising an exception, return None
        print("No pose landmarks detected in this region.")
        return None

    points = np.array([[point.x, point.y] for point in results.pose_landmarks.landmark])
    points = points[pose_indices]

    points[:, 0] = points[:, 0] * width
    points[:, 1] = points[:, 1] * height
    points = points.astype('int')

    return points
