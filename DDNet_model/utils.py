import scipy.ndimage
from scipy.spatial.distance import cdist
import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import medfilt
import copy
import pathlib


def zoom(p, target_l=64, joints_num=15, joints_dim=2):
    p_copy = copy.deepcopy(p)
    l = p_copy.shape[0]
    p_new = np.empty([target_l, joints_num, joints_dim])
    for m in range(joints_num):
        for n in range(joints_dim):
            p_copy[:, m, n] = medfilt(p_copy[:, m, n], 3)
            p_new[:, m, n] = scipy.ndimage.zoom(p_copy[:, m, n], target_l / l)[:target_l]
    return p_new


def norm_scale(x):
    return (x - np.mean(x)) / np.mean(x)


def get_CG(p, C):
    M = []
    iu = np.triu_indices(C.joint_n, 1)
    for f in range(C.frame_l):
        d_m = cdist(p[f], p[f], "euclidean")
        d_m = d_m[iu]
        M.append(d_m)
    M = np.stack(M)
    M = norm_scale(M)
    return M


def poses_diff(x):
    # Update to handle input of shape (batch_size, frame_length, joint_dims)
    batch_size, frame_length, joints_dims = x.shape
    x = x[:, 1:, ...] - x[:, :-1, ...]
    x = x.permute(0, 2, 1)  # Change to (batch_size, joint_dims, frame_length - 1)
    x = F.interpolate(x, size=frame_length, align_corners=False, mode="linear")
    x = x.permute(0, 2, 1)  # Change back to (batch_size, frame_length, joint_dims)
    return x


def poses_motion(P):
    # Slow motion (frame-to-frame differences)
    P_diff_slow = poses_diff(P)

    # Fast motion (use every second frame)
    P_fast = P[:, ::2, :]  # Use only three indices to handle (batch_size, frame_length, joint_dims)
    P_diff_fast = poses_diff(P_fast)

    # Flatten the pose differences to match LSTM input requirements
    P_diff_slow = torch.flatten(P_diff_slow, start_dim=2)
    P_diff_fast = torch.flatten(P_diff_fast, start_dim=2)

    return P_diff_slow, P_diff_fast


def makedir(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
