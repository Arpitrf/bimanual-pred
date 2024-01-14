import open3d as o3d
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


path = './data/bimanual/tiago_tissue_roll_1'
save_folder = './data/bimanual/tissue2'
os.makedirs(save_folder, exist_ok=True)


def visualize_screw_axis(s_hat, q=np.array([0.7, 0.0, 0.72])):
    # Define the starting point of the line
    start_point = q

    # Define the direction vector of the line
    # direction_vector = np.array([0.0, 1.0, 0.0])
    direction_vector = s_hat

    # Normalize the direction vector to make it a unit vector
    direction_vector /= np.linalg.norm(direction_vector)

    # Define the length of the line
    line_length = 1

    # Calculate the end point of the line
    end_point = start_point + line_length * direction_vector

    print("end_point: ", end_point)

    # Create a LineSet
    line_set = o3d.geometry.LineSet()

    # Add the line to the LineSet
    line_set.points = o3d.utility.Vector3dVector(np.vstack([start_point, end_point]))
    line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
    return line_set

def translation_augmentation(pcd, disp=np.array([0.05, 0.05, 0.0])):
    points = np.array(pcd.points)
    # making dimenson (n,3) -> (n,4) by appending 1 to each point
    ones_column = np.ones((points.shape[0], 1), dtype=points.dtype)
    points = np.append(points, ones_column, axis=1)
    translation_matrix = np.array([
        [1, 0, 0, disp[0]],
        [0, 1, 0, disp[1]],
        [0, 0, 1, disp[2]],
        [0, 0, 0, 1]
    ])
    # Apply the rotation to the point cloud
    transformed_points = np.dot(points, translation_matrix.T)
    transformed_pcd = o3d.geometry.PointCloud()
    transformed_pcd.points = o3d.utility.Vector3dVector(transformed_points[:, :3])
    transformed_pcd.colors = pcd.colors
    
    return transformed_pcd

def rotation_augmentation(pcd, gt, angle_radians=0.1):
    points = np.array(pcd.points)
    # Define the 3D rotation matrix around the z-axis
    rotation_matrix = np.array([[np.cos(angle_radians), -np.sin(angle_radians), 0],
                                [np.sin(angle_radians), np.cos(angle_radians), 0],
                                [0, 0, 1]])
    # Apply the rotation to the point cloud
    transformed_points = np.dot(points, rotation_matrix.T)
    transformed_pcd = o3d.geometry.PointCloud()
    transformed_pcd.points = o3d.utility.Vector3dVector(transformed_points)
    transformed_pcd.colors = pcd.colors

    # Apply the rotation to the gt
    transformed_gt = {}
    for k, v in gt.items():
        v_new = np.dot(v, rotation_matrix.T)
        transformed_gt[k] = v_new
    
    return transformed_pcd, transformed_gt


intr = np.array([
    [523.9963414139355, 0.0, 328.83202929614686],
    [0.0, 524.4907272320442, 237.83703502879925],
    [0.0, 0.0, 1.0]
])

# Load RGB-D
color_img = cv2.imread(f'{path}/color_img/0036.jpg')
color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
height = color_img.shape[0]
width = color_img.shape[1]
with open(f'{path}/depth/0036.pickle', 'rb') as handle:
    depth_img = pickle.load(handle)

# Load extrinsic
with open(f'{path}/extrinsic.pickle', 'rb') as handle:
    extr = np.array(pickle.load(handle))
print("extr: ", extr)

# Load object mask
with open(f'{path}/masks/0036.pickle', 'rb') as handle:
    obj_masks = np.array(pickle.load(handle))
    print("obj_masks: ", len(obj_masks))

# Load contact regions for left and right hands
with open(f'{path}/contact_regions_tissue.pickle', 'rb') as handle:
    contact_regions = pickle.load(handle)
    contact_mask_left = contact_regions['left_mask']
    contact_mask_right = contact_regions['right_mask']

# Load ground truth screw axis
with open(f'{path}/axis.pickle', 'rb') as handle:
    axis = pickle.load(handle)
    s_hat = axis['s_hat']
    q = axis['q']
    print("s_hat, q: ", s_hat, q)

# ----------------- Get the pcd of just the object-------------------------
points = []
colors = []
left_contact_arr = []
right_contact_arr = []
contact_arr = []
for x in range(height):
    for y in range(width):
        keep_point = False
        for mask in obj_masks:
            if mask[x, y]:
                keep_point = True
                break
        if not keep_point:
            continue           
        depth_value = depth_img[x, y]
        rgb_value = np.array(color_img[x, y]).astype(np.float64)
        rgb_value /= 255.0

        click_z = depth_value
        click_x = (y-intr[0, 2]) * \
            click_z/intr[0, 0]
        click_y = (x-intr[1, 2]) * \
            click_z/intr[1, 1]

        # 3d point in camera coordinates
        point_cam = np.asarray([click_x, click_y, click_z])
        point_cam /= 1000
        point_cam = np.append(point_cam, 1.0)
        point_world = np.dot(extr, point_cam)
        points.append(point_world[:3])
        
        if contact_mask_left[x, y]:
            left_contact_arr.append(1)
            right_contact_arr.append(0)
            contact_arr.append(1)
            # colors.append([0.5, 0.5, 0.5])
        elif contact_mask_right[x, y]:
            right_contact_arr.append(1)
            left_contact_arr.append(0)
            contact_arr.append(2)
            # colors.append([0, 0, 0])
        else:
            left_contact_arr.append(0)
            right_contact_arr.append(0)
            contact_arr.append(0)
            # colors.append(rgb_value)
        
        colors.append(rgb_value)

points = np.array(points)     
colors = np.array(colors)   
left_contact_arr = np.array(left_contact_arr)
right_contact_arr = np.array(right_contact_arr)
contact_arr = np.array(contact_arr)

# # Testing if seg_labels is correct
# unique_values, counts = np.unique(contact_arr, return_counts=True)
# for value, count in zip(unique_values, counts):
#     print(f"{value} occurs {count} times")

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)
print("Before cleaning pcd shape: ", np.array(pcd.points).shape, contact_arr.shape)

# Clean up pcd
cl, ind = pcd.remove_radius_outlier(nb_points=500, radius=0.05)
# cl, ind = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)

# TODO: Downsample pcd

# select by index
pcd = pcd.select_by_index(ind)

left_contact_arr = left_contact_arr[ind]
right_contact_arr = right_contact_arr[ind]
contact_arr = contact_arr[ind]
print("After cleaning pcd shape ", np.array(pcd.points).shape, contact_arr.shape)
# ------------------------------------------------------------------------

# Visualize pcds
# line_set = visualize_screw_axis()
# o3d.visualization.draw_geometries([pcd, line_set])
# o3d.visualization.draw_geometries([pcd, transformed_pcd])

# GT scew axis. For tissue task only s_hat is needed
gt = {
    's_hat': s_hat
}

counter = 0

# Generate translation augmentations for a given datapoint
for _ in range(2):
    disp_x = np.random.uniform(-0.2, 0.2)
    disp_y = np.random.uniform(-0.2, 0.2)
    disp_z = 0.0
    disp = np.array([disp_x, disp_y, disp_z])
    transformed_pcd = translation_augmentation(pcd, disp)
    # Obtain normals
    transformed_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    points = np.array(transformed_pcd.points)
    normals = np.array(transformed_pcd.normals)
    left_contact_arr_exp = left_contact_arr[:, np.newaxis]
    right_contact_arr_exp = right_contact_arr[:, np.newaxis]
    contact_arr_exp = contact_arr[:, np.newaxis]
    combined_arr = np.concatenate((points, normals, contact_arr_exp), axis=1)
    save_dict = {}
    save_dict['points'] = points
    save_dict['normals'] = normals
    save_dict['contact_arr_exp'] = contact_arr_exp    
    save_dict['gt'] = gt
    
    line_set = visualize_screw_axis(gt['s_hat'])
    o3d.visualization.draw_geometries([pcd, transformed_pcd, line_set])
    
    # Save to disk
    with open(f'{save_folder}/{counter}.pickle', 'wb') as handle:
        pickle.dump(combined_arr, handle, protocol=pickle.HIGHEST_PROTOCOL)
    np.savetxt(f"{save_folder}/{counter}.csv", combined_arr, delimiter=",")
    counter += 1

# Generate rotation augmentations for a given datapoint
for _ in range(2):
    # (-45, 45) degrees
    angle_radians = np.random.uniform(-0.78, 0.78)
    transformed_pcd, transformed_gt = rotation_augmentation(pcd, gt, angle_radians)
    # Obtain normals
    transformed_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    points = np.array(transformed_pcd.points)
    normals = np.array(transformed_pcd.normals)
    contact_arr_exp = contact_arr[:, np.newaxis]
    combined_arr = np.concatenate((points, normals, contact_arr_exp), axis=1)
    save_dict = {}
    save_dict['points'] = points
    save_dict['normals'] = normals
    save_dict['contact_arr_exp'] = contact_arr_exp    
    save_dict['gt'] = transformed_gt
    
    # Visualzie the original pcd, transformed pcd and the transformed axis
    line_set = visualize_screw_axis(transformed_gt['s_hat'])
    o3d.visualization.draw_geometries([pcd, transformed_pcd, line_set])

    # Save to disk
    with open(f'{save_folder}/{counter}.pickle', 'wb') as handle:
        pickle.dump(combined_arr, handle, protocol=pickle.HIGHEST_PROTOCOL)
    np.savetxt(f"{save_folder}/{counter}.csv", combined_arr, delimiter=",")
    counter += 1

