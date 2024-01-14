# *_*coding:utf-8 *_*
import os
import json
import warnings
import open3d as o3d
from pickle import load
import numpy as np
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

class PartNormalDataset(Dataset):
    def __init__(self, root = './data/bimanual/tissue', task='contact', split='train', npoints=2500, normal_channel=False, use_q=False):
        self.root = root
        self.npoints = npoints
        self.task = task
        self.split = split
        self.normal_channel = normal_channel
        self.use_q = use_q

        # load pcl from csv file
        data = np.loadtxt(os.path.join(self.root, '0.csv'), delimiter=',').astype(np.float32) # dim nx7: [x,y,z,nx,ny,nz,seg]
        axis = np.loadtxt(os.path.join(self.root, '0_axis.csv'), delimiter=',').astype(np.float32) # dim 2x3: [s_hat, q]
        self.axis = axis

        # set xyz, nml, seg
        if not self.normal_channel:
            self.point_set = data[:, 0:3]
        else:
            self.point_set = data[:, 0:6]
        self.seg = data[:, -1].astype(np.int32)
        self.point_set[:, 0:3] = pc_normalize(self.point_set[:, 0:3])

        # set augmentation ranges
        self.disp_range = [0.2, 0.2, 0.0]  # 20 cm
        self.angle_range = 0.78  # 45 degrees

    def __getitem__(self, idx):

        # downsample
        choice = np.random.choice(len(self.seg), self.npoints, replace=True)
        pcl = self.point_set[choice, :]
        seg = self.seg[choice]

        # translation
        if self.split == 'train':
            # uniform random sampling
            disp_x = np.random.uniform(-self.disp_range[0], self.disp_range[0])
            disp_y = np.random.uniform(-self.disp_range[1], self.disp_range[1])
        elif self.split == 'val':
            # fixed values based on idx
            disp_x = -self.disp_range[0] + 2*self.disp_range[0]*idx/self.__len__()
            disp_y = -self.disp_range[1] + 2*self.disp_range[1]*idx/self.__len__()
        disp_z = 0.0
        disp = np.array([disp_x, disp_y, disp_z])
        pcl[:,:3], axis = self.translation_augmentation(pcl[:,:3], self.axis, disp)

        # rotation
        if self.split == 'train':
            # uniform random sampling
            self.angle_radians = np.random.uniform(-self.angle_range, self.angle_range)
        elif self.split == 'val':
            # fixed values based on idx
            self.angle_radians = -self.angle_range + 2*self.angle_range*idx/self.__len__()
        pcl[:,:3], axis = self.rotation_augmentation(pcl[:,:3], self.axis, self.angle_radians)

        # Obtain normals
        if self.normal_channel:
            pcl_o3d = o3d.geometry.PointCloud()
            pcl_o3d.points = o3d.utility.Vector3dVector(pcl[:,:3])
            pcl_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            pcl[:,3:6] = np.array(pcl_o3d.normals)

        if self.task == 'contact':
            return pcl, seg
        elif self.task == 'axis':
            if self.use_q:
                return pcl, axis.flatten()
            else:
                return pcl, axis[0]
        else:
            raise ValueError('Task not recognized.')

    def __len__(self):
        if self.split == 'train':
            return 16
        elif self.split == 'val':
            return 16
    
    def translation_augmentation(self, points, gt, disp=np.array([0.05, 0.05, 0.0])):
        translation_matrix = np.array([
            [1, 0, 0, disp[0]],
            [0, 1, 0, disp[1]],
            [0, 0, 1, disp[2]],
            [0, 0, 0, 1]
        ])
        # Apply the translation to the point cloud
        # making dimenson (n,3) -> (n,4) by appending 1 to each point
        ones_column = np.ones((points.shape[0], 1), dtype=points.dtype)
        points = np.append(points, ones_column, axis=1)
        transformed_points = np.dot(points, translation_matrix.T)
        # Apply the translation to the gt
        ones_column = np.ones((gt.shape[0], 1), dtype=points.dtype)
        gt = np.append(gt, ones_column, axis=1)
        transformed_gt = np.dot(gt, translation_matrix.T)
        
        return transformed_points[:,:3], transformed_gt

    def rotation_augmentation(self, points, gt, angle_radians=0.1):
        # Define the 3D rotation matrix around the z-axis
        rotation_matrix = np.array([[np.cos(angle_radians), -np.sin(angle_radians), 0],
                                    [np.sin(angle_radians), np.cos(angle_radians), 0],
                                    [0, 0, 1]])
        # Apply the rotation to the point cloud
        transformed_points = np.dot(points, rotation_matrix.T)
        # Apply the rotation to the gt
        transformed_gt = np.dot(gt, rotation_matrix.T)
        
        return transformed_points, transformed_gt
