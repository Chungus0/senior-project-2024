#! /usr/bin/env python
#! coding:utf-8

import scipy.ndimage.interpolation as inter
from scipy.spatial.distance import cdist
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pathlib
import copy
from scipy.signal import medfilt


# Resize frames to target length
def zoom(p, target_l=64, joints_num=15, joints_dim=2):
    """
    Resizes the pose sequence to have a fixed number of frames.
    Args:
        p: Input array of shape (frames, joints_num, joints_dim).
        target_l: Target number of frames.
        joints_num: Number of joints.
        joints_dim: Dimension of joints (x, y).
    Returns:
        np.array: Resized pose sequence with shape (target_l, joints_num, joints_dim).
    """
    p_copy = copy.deepcopy(p)
    l = p_copy.shape[0]
    p_new = np.empty([target_l, joints_num, joints_dim])
    for m in range(joints_num):
        for n in range(joints_dim):
            p_copy[:, m, n] = medfilt(p_copy[:, m, n], 3)  # Smooth the sequence
            p_new[:, m, n] = inter.zoom(p_copy[:, m, n], target_l / l)[:target_l]
    return p_new


# Calculate the Joint-Centered Distance (JCD) feature
def norm_scale(x):
    """Normalizes by centering around zero mean and scaling by mean value."""
    return (x - np.mean(x)) / np.mean(x)


def get_CG(p, C):
    """
    Generates the JCD matrix (distance matrix of joint positions).
    Args:
        p: Pose array of shape (frame_l, joint_n, joint_d).
        C: Configuration object with attributes `frame_l` and `joint_n`.
    Returns:
        np.array: Normalized distance matrix of shape (frame_l, feat_d).
    """
    M = []
    iu = np.triu_indices(C.joint_n, 1)  # Upper triangle indices without diagonal
    for f in range(C.frame_l):
        d_m = cdist(p[f], p[f], "euclidean")
        d_m = d_m[iu]  # Extract upper triangle, flatten into vector
        M.append(d_m)
    M = np.stack(M)
    M = norm_scale(M)  # Normalize the distance matrix
    return M


def poses_diff(x):
    """
    Calculates frame-to-frame differences in the pose sequence.
    Args:
        x: Pose array of shape (batch, channels, joints, dims).
    Returns:
        torch.Tensor: Frame-to-frame difference with bilinear interpolation.
    """
    _, H, W, _ = x.shape
    x = x[:, 1:, ...] - x[:, :-1, ...]
    x = x.permute(0, 3, 1, 2)  # (batch, dim, frames, joints)
    x = F.interpolate(x, size=(H, W), align_corners=False, mode="bilinear")
    x = x.permute(0, 2, 3, 1)  # (batch, frames, joints, dim)
    return x


def poses_motion(P):
    """
    Generates both slow and fast motion representations from pose sequence.
    Args:
        P: Pose array of shape (batch, frames, joints, dims).
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Slow and fast pose motions.
    """
    P_diff_slow = poses_diff(P)
    P_diff_slow = torch.flatten(P_diff_slow, start_dim=2)  # Flatten for each frame
    P_fast = P[:, ::2, :, :]  # Downsample by 2 for fast motion
    P_diff_fast = poses_diff(P_fast)
    P_diff_fast = torch.flatten(P_diff_fast, start_dim=2)
    return P_diff_slow, P_diff_fast  # Returns slow and fast motion


def makedir(path):
    """Creates directory if it doesn't exist."""
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
