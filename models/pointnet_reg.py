"""
Train PointNet to predict screw axis
"""

import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from pointnet_utils import PointNetEncoder, feature_transform_reguliarzer

class get_model(nn.Module):
    def __init__(self, k=6, normal_channel=True):
        super(get_model, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x, trans_feat

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001, axis_loss_scale=1.0):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale
        self.axis_loss_scale = axis_loss_scale

    def forward(self, pred, target, trans_feat):
        axis_loss = F.mse_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)
        total_loss = self.axis_loss_scale * axis_loss + self.mat_diff_loss_scale * mat_diff_loss
        loss_dict = {'total': total_loss, 
                     'axis': axis_loss, 
                     'mat_diff': mat_diff_loss}
        return loss_dict