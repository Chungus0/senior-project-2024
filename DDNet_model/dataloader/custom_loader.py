#! /usr/bin/env python
#! coding:utf-8

from tqdm import tqdm
import numpy as np
from sklearn import preprocessing
import pickle
from pathlib import Path
from utils import get_CG, zoom  # Make sure utils contains updated `get_CG` and `zoom`


# Load the multi-class dataset (squat, standing, sitting)
def load_multiclass_data(train_path="data/Custom/GT_train_multi.pkl", test_path="data/Custom/GT_test_multi.pkl"):
    """
    Load training and test data from pickle files and fit a LabelEncoder on the combined labels.

    Args:
        train_path (str): Path to the training dataset pickle file.
        test_path (str): Path to the testing dataset pickle file.

    Returns:
        tuple: Training data, testing data, and the fitted LabelEncoder.
    """
    # Load the training and testing data from the pickle files
    with open(train_path, "rb") as f:
        Train = pickle.load(f)
    with open(test_path, "rb") as f:
        Test = pickle.load(f)

    # Label encoder: combine both training and test labels to fit
    combined_labels = Train["label"] + Test["label"]
    le = preprocessing.LabelEncoder()
    le.fit(combined_labels)

    print("Loading Multi-Class Dataset (Squat, Standing, Sitting)")
    return Train, Test, le


# Configuration for the multi-class dataset
class MultiClassConfig:
    def __init__(self):
        self.frame_l = 64  # Fixed length of frames to standardize input size
        self.joint_n = 15  # Number of joints
        self.joint_d = 2  # Dimension of joints (x, y)
        self.clc_num = 3  # Number of classes (squat, standing, sitting)
        self.feat_d = 105  # Feature dimension (upper triangle distance matrix of 15 joints)
        self.filters = 64  # Number of filters for the network


# Generate dataset for the multi-class action dataset
def MultiClassDataGenerator(T, C, le):
    """
    Generates training data for the multi-class action dataset.

    Args:
        T (dict): Dataset containing 'pose' and 'label' keys.
        C (MultiClassConfig): Configuration object.
        le (LabelEncoder): Fitted LabelEncoder.

    Returns:
        tuple: JCD features (X_0), pose keypoints (X_1), and labels (Y).
    """
    X_0 = []  # Stores JCD features (upper triangle matrix of distances)
    X_1 = []  # Stores pose keypoints
    Y = []  # Stores labels (squat, standing, sitting)

    # Convert the labels using the LabelEncoder
    labels = le.transform(T["label"])

    # Iterate over all the pose data in the dataset
    for i in tqdm(range(len(T["pose"]))):
        p = np.copy(T["pose"][i])  # Get the pose data for each frame (frame, joints, coords)

        # Standardize the pose data to have a fixed frame length (64 frames)
        p = zoom(p, target_l=C.frame_l, joints_num=C.joint_n, joints_dim=C.joint_d)

        # Compute the JCD (Joint-Centered Distance) matrix for the pose
        M = get_CG(p, C)

        # Append the pose and JCD features to their respective lists
        X_0.append(M)  # JCD features
        X_1.append(p)  # Pose keypoints
        Y.append(labels[i])  # Corresponding label

    # Convert the lists to numpy arrays
    X_0 = np.stack(X_0)
    X_1 = np.stack(X_1)
    Y = np.stack(Y)

    return X_0, X_1, Y  # Return the generated dataset
