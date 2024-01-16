"""
Train PointNet to predict screw axis
"""

import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from pointnet_utils import PointNetEncoder, feature_transform_reguliarzer

class get_model(nn.Module):
    def __init__(self, k=6, normal_channel=True, use_q=False, use_s=False):
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
        self.use_q = use_q
        self.use_s = use_s

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        # x = F.log_softmax(x, dim=1)
        # ---- normalize s_hat -----
        # print(x, torch.norm(x[:, :3], dim=1))
        if self.use_s:
            norm = torch.unsqueeze(torch.norm(x[:, :3], dim=1), 1)
            pred_s_norm = x[:, :3] / norm   # (16, 3)
            pred_q = x[:, 3:]               # (16,3)
            out = torch.cat((pred_s_norm,  pred_q), 1)
        else:
            out = x
        # print(out, torch.norm(out[:, :3], dim=1))
        # print("out shape: ", out.shape)
        # --------------------------
        return out, trans_feat

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001, axis_loss_scale=1.0):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale
        self.axis_loss_scale = axis_loss_scale

    def forward(self, pred, target, trans_feat):
        print("pred, target: ", pred.shape, target.shape)
        axis_loss = F.mse_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)
        # change later
        total_loss = self.axis_loss_scale * axis_loss + self.mat_diff_loss_scale * mat_diff_loss
        print("total_loss: ", total_loss)
        # total_loss = self.axis_loss_scale * axis_loss 
        loss_dict = {'total': total_loss, 
                     'axis': axis_loss, 
                     'mat_diff': mat_diff_loss}
        return loss_dict