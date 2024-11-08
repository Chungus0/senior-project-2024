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

# Initialize MediaPipe pose estimator and drawing utilities
mp_pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=2)
mp_drawing = mp.solutions.drawing_utils  # For drawing the pose skeleton on video frames


def extract_poses_from_video(video_path, target_frame_length, output_path=None, augment=False):
    """
    Extracts poses from the input video using MediaPipe Pose Estimation.

    Args:
        video_path (str): Path to the video file.
        target_frame_length (int): Target length to which all video sequences will be resized.
        output_path (str, optional): Path to save the video with skeleton overlaid. Defaults to None.
        augment (bool): If True, applies random noise for data augmentation.

    Returns:
        np.array: An array of extracted 2D pose coordinates for each frame (15 joints).
    """
    logging.info(f"Processing video: {video_path}")

    # Open the video file using OpenCV
    cap = cv2.VideoCapture(video_path)
    poses = []

    # If output_path is provided, prepare to save the video with overlaid skeleton
    if output_path is not None:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Set codec for saving the video
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Loop through each frame of the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Exit if no more frames

        # Convert the frame from BGR (OpenCV) to RGB (required by MediaPipe)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform pose detection
        result = mp_pose.process(frame_rgb)

        # If pose landmarks are detected, store the 2D coordinates (x, y) for the first 15 joints
        if result.pose_landmarks:
            pose = np.array([[landmark.x, landmark.y] for landmark in result.pose_landmarks.landmark])
            pose = pose[:15, :2]  # Use only 15 key joints for this application

            # Apply random noise for data augmentation
            if augment:
                pose += np.random.normal(0, 0.01, pose.shape)

            poses.append(pose)

            # Draw the pose skeleton on the frame for visualization purposes
            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
        else:
            # If no pose detected, append a blank pose (zeros)
            poses.append(np.zeros((15, 2)))

        # If output video is being saved, write the current frame with the drawn skeleton
        if output_path is not None:
            out.write(frame)

    # Release the video capture and writer objects
    cap.release()
    if output_path is not None:
        out.release()

    logging.info(f"Completed pose extraction for video: {video_path}")

    # Resize poses to target frame length
    poses = resize_sequence(poses, target_frame_length)

    # Return the list of extracted poses as a NumPy array
    return np.array(poses)


def resize_sequence(poses, target_length):
    """
    Resizes the pose sequence to a target length using linear interpolation.

    Args:
        poses (list of np.array): List of pose arrays.
        target_length (int): Desired length of the sequence.

    Returns:
        list of np.array: Resized list of pose arrays.
    """
    original_length = len(poses)
    new_indices = np.linspace(0, original_length - 1, target_length)
    resized_poses = [poses[int(idx)] for idx in new_indices]
    return resized_poses


def process_videos_from_folder(folder_path, label, target_frame_length, output_folder=None, augment=False):
    """
    Processes all videos in a folder, extracts poses, and stores them in a dataset format.

    Args:
        folder_path (str): Path to the folder containing video files.
        label (int): Label for the action (e.g., 0 for sitting, 1 for standing, 2 for squatting).
        target_frame_length (int): Target length to which all video sequences will be resized.
        output_folder (str, optional): Folder to save videos with skeleton overlaid. Defaults to None.
        augment (bool): If True, applies random noise for data augmentation.

    Returns:
        dict: A dataset containing extracted poses and corresponding labels.
    """
    dataset = {"pose": [], "label": []}  # To store pose data for each video and labels

    # List all mp4 video files in the folder
    video_files = [f for f in os.listdir(folder_path) if f.endswith(".mp4")]

    logging.info(f"Found {len(video_files)} videos in folder: {folder_path}")

    # Iterate over all video files in the folder
    for video_file in tqdm(video_files):
        video_path = os.path.join(folder_path, video_file)
        output_path = None

        # If output_folder is specified, set up the path for saving the skeleton-overlaid video
        if output_folder:
            output_path = os.path.join(output_folder, f"skeleton_{video_file}")
            if not os.path.exists(output_folder):
                os.makedirs(output_folder, exist_ok=True)  # Create the output folder if it doesn't exist

        # Extract poses from the video and save the result to the dataset
        poses = extract_poses_from_video(video_path, target_frame_length, output_path, augment=augment)
        dataset["pose"].append(poses)
        dataset["label"].append(label)

    logging.info(f"Completed processing for folder: {folder_path}")
    return dataset


def save_dataset(dataset, file_path):
    """
    Save the dataset to a pickle file.

    Args:
        dataset (dict): Dictionary containing the dataset.
        file_path (str): Path to the file where the dataset will be saved.
    """
    with open(file_path, "wb") as f:
        pickle.dump(dataset, f)
    logging.info(f"Dataset saved to {file_path}")


# Paths to folders containing action videos
squat_folder = "squat_videos"
standing_folder = "standing_videos"
sitting_folder = "sitting_videos"

# Optional output folders for saving skeleton-overlaid videos
squat_output_folder = "skeleton_squat_videos"
standing_output_folder = "skeleton_standing_videos"
sitting_output_folder = "skeleton_sitting_videos"

# Target frame length for resizing sequences
target_frame_length = 64

logging.info("Starting dataset preparation...")

# Process each action category with unique labels
squat_data = process_videos_from_folder(
    squat_folder, label=0, target_frame_length=target_frame_length, output_folder=squat_output_folder, augment=True
)
standing_data = process_videos_from_folder(
    standing_folder,
    label=1,
    target_frame_length=target_frame_length,
    output_folder=standing_output_folder,
    augment=True,
)
sitting_data = process_videos_from_folder(
    sitting_folder, label=2, target_frame_length=target_frame_length, output_folder=sitting_output_folder, augment=True
)

# Combine datasets from all action classes
combined_data = {
    "pose": squat_data["pose"] + standing_data["pose"] + sitting_data["pose"],
    "label": squat_data["label"] + standing_data["label"] + sitting_data["label"],
}

# Shuffle the combined dataset to randomize the data order
combined_indices = list(range(len(combined_data["pose"])))
random.shuffle(combined_indices)
combined_data["pose"] = [combined_data["pose"][i] for i in combined_indices]
combined_data["label"] = [combined_data["label"][i] for i in combined_indices]

# Split the combined dataset into training and testing sets (80/20 split)
split_index = int(len(combined_data["pose"]) * 0.8)
train_data = {"pose": combined_data["pose"][:split_index], "label": combined_data["label"][:split_index]}
test_data = {"pose": combined_data["pose"][split_index:], "label": combined_data["label"][split_index:]}

# Save the training and testing sets to pickle files
save_dataset(train_data, "GT_train_multi.pkl")
save_dataset(test_data, "GT_test_multi.pkl")

logging.info("Dataset preparation completed successfully!")
