import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import poses_motion


class c1D(nn.Module):
    def __init__(self, input_channels, input_dims, filters, kernel):
        super(c1D, self).__init__()
        self.cut_last_element = kernel % 2 == 0
        self.padding = math.ceil((kernel - 1) / 2)
        self.conv1 = nn.Conv1d(input_dims, filters, kernel, bias=False, padding=self.padding)
        self.bn = nn.BatchNorm1d(num_features=filters)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        if self.cut_last_element:
            output = self.conv1(x)[:, :, :-1]
        else:
            output = self.conv1(x)
        output = self.bn(output)
        output = F.leaky_relu(output, 0.2, inplace=True)
        return output.permute(0, 2, 1)


class block(nn.Module):
    def __init__(self, input_channels, input_dims, filters, kernel):
        super(block, self).__init__()
        self.c1D1 = c1D(input_channels, input_dims, filters, kernel)
        self.c1D2 = c1D(input_channels, filters, filters, kernel)

    def forward(self, x):
        output = self.c1D1(x)
        output = self.c1D2(output)
        return output


class d1D(nn.Module):
    def __init__(self, input_dims, filters):
        super(d1D, self).__init__()
        self.linear = nn.Linear(input_dims, filters)
        self.bn = nn.BatchNorm1d(num_features=filters)

    def forward(self, x):
        output = self.linear(x)
        output = self.bn(output)
        return F.leaky_relu(output, 0.2, inplace=True)


class spatialDropout1D(nn.Module):
    def __init__(self, p):
        super(spatialDropout1D, self).__init__()
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        return self.dropout(x.permute(0, 2, 1)).permute(0, 2, 1)


class DDNet_MultiClass(nn.Module):
    def __init__(self, frame_l, joint_n, joint_d, feat_d, filters, class_num):
        super(DDNet_MultiClass, self).__init__()

        self.jcd_conv1 = nn.Sequential(c1D(frame_l, feat_d, 4 * filters, 1), spatialDropout1D(0.1))
        self.jcd_conv2 = nn.Sequential(c1D(frame_l, 4 * filters, 2 * filters, 3), spatialDropout1D(0.1))
        self.jcd_conv3 = c1D(frame_l, 2 * filters, 2 * filters, 1)
        self.jcd_pool = nn.Sequential(nn.MaxPool1d(kernel_size=2), spatialDropout1D(0.1))

        self.slow_conv1 = nn.Sequential(c1D(frame_l, joint_n * joint_d, 4 * filters, 1), spatialDropout1D(0.1))
        self.slow_conv2 = nn.Sequential(c1D(frame_l, 4 * filters, 2 * filters, 3), spatialDropout1D(0.1))
        self.slow_conv3 = c1D(frame_l, 2 * filters, 2 * filters, 1)
        self.slow_pool = nn.Sequential(nn.MaxPool1d(kernel_size=2), spatialDropout1D(0.1))

        self.fast_conv1 = nn.Sequential(c1D(frame_l // 2, joint_n * joint_d, 4 * filters, 1), spatialDropout1D(0.1))
        self.fast_conv2 = nn.Sequential(c1D(frame_l // 2, 4 * filters, 2 * filters, 3), spatialDropout1D(0.1))
        self.fast_conv3 = c1D(frame_l // 2, 2 * filters, 2 * filters, 1)

        self.block1 = block(frame_l // 2, 6 * filters, 4 * filters, 3)
        self.block_pool1 = nn.Sequential(nn.MaxPool1d(kernel_size=2), spatialDropout1D(0.1))

        self.block2 = block(frame_l // 4, 4 * filters, 8 * filters, 3)
        self.block_pool2 = nn.Sequential(nn.MaxPool1d(kernel_size=2), spatialDropout1D(0.1))

        self.block3 = nn.Sequential(block(frame_l // 8, 8 * filters, 16 * filters, 3), spatialDropout1D(0.1))

        # LSTM layer for temporal dynamics
        self.lstm = nn.LSTM(
            input_size=16 * filters, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True
        )

        self.linear1 = nn.Sequential(d1D(256, 128), nn.Dropout(0.5))  # Adjusted input size for LSTM output
        self.linear2 = nn.Sequential(d1D(128, 128), nn.Dropout(0.5))
        self.linear3 = nn.Linear(128, class_num)

    def forward(self, M, P=None):
        # JCD feature processing
        x = self.jcd_conv1(M)
        x = self.jcd_conv2(x)
        x = self.jcd_conv3(x)
        x = x.permute(0, 2, 1)
        x = self.jcd_pool(x)
        x = x.permute(0, 2, 1)

        # Pose motion processing
        diff_slow, diff_fast = poses_motion(P)

        # Slow motion path
        x_d_slow = self.slow_conv1(diff_slow)
        x_d_slow = self.slow_conv2(x_d_slow)
        x_d_slow = self.slow_conv3(x_d_slow)
        x_d_slow = x_d_slow.permute(0, 2, 1)
        x_d_slow = self.slow_pool(x_d_slow)
        x_d_slow = x_d_slow.permute(0, 2, 1)

        # Fast motion path
        x_d_fast = self.fast_conv1(diff_fast)
        x_d_fast = self.fast_conv2(x_d_fast)
        x_d_fast = self.fast_conv3(x_d_fast)

        # Concatenate JCD, slow motion, and fast motion outputs
        x = torch.cat((x, x_d_slow, x_d_fast), dim=2)

        # Pass through block layers and max-pooling
        x = self.block1(x)
        x = x.permute(0, 2, 1)
        x = self.block_pool1(x)
        x = x.permute(0, 2, 1)

        x = self.block2(x)
        x = x.permute(0, 2, 1)
        x = self.block_pool2(x)
        x = x.permute(0, 2, 1)

        x = self.block3(x)
        x, _ = self.lstm(x)  # Use the LSTM and ignore the hidden state output
        x = torch.max(x, dim=1).values

        # Fully connected layers
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)  # Final output for classification

        return x  # Only return the final classification output
