from tqdm import tqdm
import numpy as np
from sklearn import preprocessing
import pickle
from utils import get_CG, zoom


def load_multiclass_data(train_path="data/Custom/GT_train_multi.pkl", test_path="data/Custom/GT_test_multi.pkl"):
    with open(train_path, "rb") as f:
        Train = pickle.load(f)
    with open(test_path, "rb") as f:
        Test = pickle.load(f)
    combined_labels = Train["label"] + Test["label"]
    le = preprocessing.LabelEncoder()
    le.fit(combined_labels)
    return Train, Test, le


class MultiClassConfig:
    def __init__(self):
        self.frame_l = 64
        self.joint_n = 15
        self.joint_d = 2
        self.clc_num = 9
        self.feat_d = 105
        self.filters = 64


def MultiClassDataGenerator(T, C, le):
    X_0 = []
    X_1 = []
    Y = []
    labels = le.transform(T["label"])
    for i in tqdm(range(len(T["pose"]))):
        p = np.copy(T["pose"][i])
        p = zoom(p, target_l=C.frame_l, joints_num=C.joint_n, joints_dim=C.joint_d)
        M = get_CG(p, C)
        X_0.append(M)
        X_1.append(p)
        Y.append(labels[i])
    X_0 = np.stack(X_0)
    X_1 = np.stack(X_1)
    Y = np.stack(Y)

    # Add an extra dimension to X_1 for LSTM compatibility (batch_size, seq_len, input_size)
    X_1 = X_1.reshape(X_1.shape[0], C.frame_l, C.joint_n * C.joint_d)
    return X_0, X_1, Y
