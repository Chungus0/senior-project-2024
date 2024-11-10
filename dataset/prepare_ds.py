import cv2
import mediapipe as mp
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import os
import logging
import random

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

mp_pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=2)
mp_drawing = mp.solutions.drawing_utils


def extract_poses_from_video(video_path, target_frame_length, output_path=None, augment=False):
    logging.info(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    poses = []
    if output_path is not None:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = mp_pose.process(frame_rgb)
        if result.pose_landmarks:
            pose = np.array([[landmark.x, landmark.y] for landmark in result.pose_landmarks.landmark])
            pose = pose[:15, :2]
            if augment:
                pose += np.random.normal(0, 0.01, pose.shape)
            poses.append(pose)
            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
        else:
            poses.append(np.zeros((15, 2)))
        if output_path is not None:
            out.write(frame)
    cap.release()
    if output_path is not None:
        out.release()
    logging.info(f"Completed pose extraction for video: {video_path}")
    poses = resize_sequence(poses, target_frame_length)
    return np.array(poses)


def resize_sequence(poses, target_length):
    original_length = len(poses)
    new_indices = np.linspace(0, original_length - 1, target_length)
    resized_poses = [poses[int(idx)] for idx in new_indices]
    return resized_poses


def process_videos_from_folder(folder_path, label, target_frame_length, output_folder=None, augment=False):
    dataset = {"pose": [], "label": []}
    video_files = [f for f in os.listdir(folder_path) if f.endswith(".mp4")]
    logging.info(f"Found {len(video_files)} videos in folder: {folder_path}")
    for video_file in tqdm(video_files):
        video_path = os.path.join(folder_path, video_file)
        output_path = None
        if output_folder:
            output_path = os.path.join(output_folder, f"skeleton_{video_file}")
            if not os.path.exists(output_folder):
                os.makedirs(output_folder, exist_ok=True)
        poses = extract_poses_from_video(video_path, target_frame_length, output_path, augment=augment)
        dataset["pose"].append(poses)
        dataset["label"].append(label)
    logging.info(f"Completed processing for folder: {folder_path}")
    return dataset


def save_dataset(dataset, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(dataset, f)
    logging.info(f"Dataset saved to {file_path}")


video_folders = {
    "bend": 0,
    "M_jump": 1,
    "M_lay": 2,
    "M_run": 3,
    "M_sit": 4,
    "M_walk": 5,
    "squats": 6,
    "stand": 7,
    "wave": 8,
}

output_folders = {
    "bend": "skeleton_bend_videos",
    "M_jump": "skeleton_M_jump_videos",
    "M_lay": "skeleton_M_lay_videos",
    "M_run": "skeleton_M_run_videos",
    "M_sit": "skeleton_M_sit_videos",
    "M_walk": "skeleton_M_walk_videos",
    "squats": "skeleton_squats_videos",
    "stand": "skeleton_stand_videos",
    "wave": "skeleton_wave_videos",
}

target_frame_length = 64
logging.info("Starting dataset preparation...")
combined_data = {"pose": [], "label": []}
for action, label in video_folders.items():
    folder_path = os.path.join("videos", action)
    output_folder = output_folders[action]
    data = process_videos_from_folder(
        folder_path, label, target_frame_length, output_folder=output_folder, augment=True
    )
    combined_data["pose"].extend(data["pose"])
    combined_data["label"].extend(data["label"])
combined_indices = list(range(len(combined_data["pose"])))
random.shuffle(combined_indices)
combined_data["pose"] = [combined_data["pose"][i] for i in combined_indices]
combined_data["label"] = [combined_data["label"][i] for i in combined_indices]
split_index = int(len(combined_data["pose"]) * 0.8)
train_data = {"pose": combined_data["pose"][:split_index], "label": combined_data["label"][:split_index]}
test_data = {"pose": combined_data["pose"][split_index:], "label": combined_data["label"][split_index:]}
save_dataset(train_data, "GT_train_multi.pkl")
save_dataset(test_data, "GT_test_multi.pkl")
logging.info("Dataset preparation completed successfully!")
