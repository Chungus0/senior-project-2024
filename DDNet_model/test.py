import os
import torch
import cv2
import numpy as np
from models.DDNet_MultiClass import DDNet_MultiClass
from dataloader.custom_loader import MultiClassConfig
from utils import get_CG
import mediapipe as mp
import absl.logging
import logging
from collections import deque

# Suppress MediaPipe warnings
absl.logging.set_verbosity(absl.logging.ERROR)

# Set up logging
logging.basicConfig(filename="prediction_log.txt", level=logging.INFO, format="%(asctime)s - %(message)s")

# Load model
model_path = "experiments/model.pt"  # Change if needed
Config = MultiClassConfig()
model = DDNet_MultiClass(Config.frame_l, Config.joint_n, Config.joint_d, Config.feat_d, Config.filters, Config.clc_num)
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

# Class labels for better readability
class_labels = ["bend", "jump", "lay", "run", "sit", "walk", "squat", "stand", "wave"]

# Initialize MediaPipe pose estimator
mp_pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=2)

# Capture webcam input
cap = cv2.VideoCapture(0)

# Frame buffer for storing sequences of poses
frame_buffer = deque(maxlen=Config.frame_l)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Extract pose using MediaPipe
    result = mp_pose.process(frame_rgb)

    if result.pose_landmarks:
        pose = np.array([[landmark.x, landmark.y] for landmark in result.pose_landmarks.landmark])
        pose = pose[:15, :2]  # Use the first 15 joints (x, y)
    else:
        pose = np.zeros((15, 2))

    # Append the extracted pose to the frame buffer
    frame_buffer.append(pose)

    predicted_label = "None"

    # If we have enough frames in the buffer, make a prediction
    if len(frame_buffer) == Config.frame_l:
        poses = np.array(frame_buffer)

        # Convert poses to the format expected by the model
        poses_tensor = torch.from_numpy(poses).unsqueeze(0).type("torch.FloatTensor")
        poses_flat = poses_tensor.view(poses_tensor.shape[0], poses_tensor.shape[1], -1)

        # Compute the JCD matrix (M) using get_CG function
        M = get_CG(poses_tensor.numpy()[0], Config)
        M_tensor = torch.from_numpy(M).unsqueeze(0).type("torch.FloatTensor")

        # Predict using the model
        with torch.no_grad():
            output = model(M_tensor, poses_flat)
            predicted_class = torch.argmax(output, dim=1).item()

        predicted_label = class_labels[predicted_class]
        print(f"Predicted action: {predicted_label}")
        logging.info(f"Predicted action: {predicted_label}")

    # Display the webcam feed
    cv2.putText(frame, f"Predicted action: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Webcam Action Recognition", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
