import cv2
import mediapipe as mp
import numpy as np
import os

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils



def extract_pose_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f" Gambar tidak ditemukan: {image_path}")
        return None

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_image)

    if not results.pose_landmarks:
        print(f" Pose tidak terdeteksi di: {image_path}")
        return None

    # koor
    landmarks = []
    for lm in results.pose_landmarks.landmark:
        landmarks.append([lm.x, lm.y, lm.visibility])

    # Ubah jadi array 1D
    feature_vector = np.array(landmarks).flatten()

    return feature_vector

# contoh
if __name__ == "__main__":
    test_image = "images/thinking/thinking_01.jpg"
    features = extract_pose_features(test_image)
    if features is not None:
        print("Fitur pose berhasil diekstrak:", features.shape)