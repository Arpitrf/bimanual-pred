# *_*coding:utf-8 *_*
import os
import json
import warnings
import open3d as o3d
from pickle import load
import numpy as np
from torch.utils.data import Dataset
from visualizer.bimanual_utils import visualize_pcl_axis
warnings.filterwarnings('ignore')
import provider


def pc_normalize(pc, axis=None):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    if axis is not None:
        axis = axis - centroid
        axis = axis / m
        return pc, axis
    else:
        return pc

class PartNormalDataset(Dataset):
    def __init__(self, root = './data/bimanual/tissue', task='contact', split='train', npoints=2500, normal_channel=False, use_q=False, use_s=False, obj=None):
        self.root = root
        self.npoints = npoints
        self.task = task
        self.split = split
        self.normal_channel = normal_channel
        self.use_q = use_q
        self.use_s = use_s
        self.obj = obj

        # load pcl from csv file
        data = np.loadtxt(os.path.join(self.root, '0.csv'), delimiter=',').astype(np.float32) # dim nx7: [x,y,z,nx,ny,nz,seg]
        axis = np.loadtxt(os.path.join(self.root, '0_axis.csv'), delimiter=',').astype(np.float32) # dim 2x3: [s_hat, q]
        self.axis = axis
        # self.axis = axis[0]
        # print("self.axis: ", self.axis)

        # set xyz, nml, seg
        if not self.normal_channel:
            self.point_set = data[:, 0:3]
        else:
            self.point_set = data[:, 0:6]
        self.seg = data[:, -1].astype(np.int32)
        # normalize the pcd and also the q (s_hat is already a unit vector so no need to normalize it)
        # print("--before: ", self.axis[1])
        self.point_set[:, 0:3], self.axis[1] = pc_normalize(self.point_set[:, 0:3], self.axis[1])
        # print("--after: ", self.axis[1])

        # set augmentation ranges
        self.disp_range = [0.2, 0.2, 0.0]  # 20 cm
        if self.obj == 'bottle':
            self.angle_range = 0.15 # 45 degrees
        else:
            self.angle_range = 0.78  # 45 degrees

    def __getitem__(self, idx):

        # downsample
        choice = np.random.choice(len(self.seg), self.npoints, replace=True)
        pcl = self.point_set[choice, :]
        seg = self.seg[choice]
        axis = self.axis
        
        # pcl_temp2 = pcl.copy()
        # pcl_temp = np.expand_dims(pcl, axis=0)
        # print("pcl_temp: ", pcl_temp.shape)
        # # pcl_temp[:, :, 0:3] = provider.jitter_point_cloud(pcl_temp[:, :, 0:3])
        # pcl_temp[:, :, 0:3] = provider.random_scale_point_cloud(pcl_temp[:, :, 0:3])
        # pcl = pcl_temp[0]
        # NUM_POINT = pcl.shape[0]
        # org_colors = np.tile([0.0, 0.0, 1.0], (NUM_POINT, 1)) # blues
        # o3d_pcl = o3d.geometry.PointCloud()
        # o3d_pcl.points = o3d.utility.Vector3dVector(pcl[:, :3])
        # o3d_pcl.colors = o3d.utility.Vector3dVector(org_colors)
        # o3d_pcl2 = o3d.geometry.PointCloud()        
        # org_colors = np.tile([1.0, 0.0, 0.0], (NUM_POINT, 1)) # blue
        # o3d_pcl2.points = o3d.utility.Vector3dVector(pcl_temp2[:, :3])
        # o3d_pcl2.colors = o3d.utility.Vector3dVector(org_colors)
        # o3d.visualization.draw_geometries([o3d_pcl, o3d_pcl2])

        # translation
        if self.split == 'train':
            # uniform random sampling
            disp_x = np.random.uniform(-self.disp_range[0], self.disp_range[0])
            disp_y = np.random.uniform(-self.disp_range[1], self.disp_range[1])
        elif self.split == 'val':
            # fixed values based on idx
            disp_x = -self.disp_range[0] + 2*self.disp_range[0]*idx/self.__len__()
            disp_y = -self.disp_range[1] + 2*self.disp_range[1]*idx/self.__len__()
            # print("disp_x, disp_y,: ", disp_x, disp_y)
        disp_z = 0.0
        disp = np.array([disp_x, disp_y, disp_z])
        pcl[:,:3], axis = self.translation_augmentation(pcl[:,:3], axis, disp)

        # rotation
        if self.split == 'train':
            # uniform random sampling
            self.angle_radians = np.random.uniform(-self.angle_range, self.angle_range)
        elif self.split == 'val':
            # fixed values based on idx
            self.angle_radians = -self.angle_range + 2*self.angle_range*idx/self.__len__()
        pcl[:,:3], axis = self.rotation_augmentation(pcl[:,:3], axis, self.angle_radians)

        # print("translation and rotation aug applied: ", disp, self.angle_radians)
        # if self.split == 'train':
        #     visualize_pcl_axis([axis], pcl.shape[0], pcl[:, :3], savepath='/home/arpit/test_projects/bimanual_predictor/temp.png', use_q=self.use_q)

        # random scaling
        rand_val = np.random.uniform(0.0, 1.0)
        if rand_val > 0.75:
            pcl_temp = np.expand_dims(pcl, axis=0)
            pcl_temp[:, :, 0:3] = provider.random_scale_point_cloud(pcl_temp[:, :, 0:3])
            pcl = pcl_temp[0]
 
        # Obtain normals
        if self.normal_channel:
            pcl_o3d = o3d.geometry.PointCloud()
            pcl_o3d.points = o3d.utility.Vector3dVector(pcl[:,:3])
            pcl_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            pcl[:,3:6] = np.array(pcl_o3d.normals)

        if self.task == 'contact':
            return pcl, seg
        elif self.task == 'axis':
            if self.use_q and self.use_s:
                return pcl, axis.flatten()
            if self.use_q:
                return pcl, axis[1]
            else:
                return pcl, axis[0]
                return pcl, axis
        else:
            raise ValueError('Task not recognized.')

    def __len__(self):
        if self.split == 'train':
            return 16
        elif self.split == 'val':
            return 16
    
    def translation_augmentation(self, points, axis, disp=np.array([0.05, 0.05, 0.0])):
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
        transformed_axis = []
        for i in range(len(axis)):
            if i == 0:
                transformed_axis.append(axis[0])
            else:
                q = np.array([axis[i][0], axis[i][1], axis[i][2], 1.0])
                q_new = np.dot(q, translation_matrix.T)
                transformed_axis.append(q_new[:3])
        
        return transformed_points[:,:3], transformed_axis

    def rotation_augmentation(self, points, axis, angle_radians=0.1):
        # Define the 3D rotation matrix around the z-axis
        rotation_matrix = np.array([[np.cos(angle_radians), -np.sin(angle_radians), 0],
                                    [np.sin(angle_radians), np.cos(angle_radians), 0],
                                    [0, 0, 1]])
        # Apply the rotation to the point cloud
        transformed_points = np.dot(points, rotation_matrix.T)
        # Apply the rotation to the gt
        transformed_axis = np.dot(axis, rotation_matrix.T)
        # print("transformed_axis norm: ", transformed_axis.shape, np.linalg.norm(transformed_axis))
        transformed_axis[0] /= np.linalg.norm(transformed_axis[0])
        # transformed_axis /= np.linalg.norm(transformed_axis)
        
        return transformed_points, transformed_axis
