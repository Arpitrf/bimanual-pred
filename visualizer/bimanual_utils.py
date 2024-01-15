"""
Visualization utilities for bimanual prediction
"""
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


def visualize_pcl_contact(seg_list, NUM_POINT, points, savepath):
    fig, ax = plt.subplots(1,len(seg_list))
    for idx, seg in enumerate(seg_list):
        org_colors = np.zeros((NUM_POINT, 3))
        org_colors[seg == 0] = [0.0, 0.0, 1.0]  # no contact: blue
        org_colors[seg == 1] = [0.0, 1.0, 0.0]  # left contact: green
        org_colors[seg == 2] = [1.0, 0.0, 0.0]  # right contact: red
        o3d_pcl = o3d.geometry.PointCloud()
        o3d_pcl.points = o3d.utility.Vector3dVector(points)
        o3d_pcl.colors = o3d.utility.Vector3dVector(org_colors)
        
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        vis.add_geometry(o3d_pcl)
        vis.update_geometry(o3d_pcl)
        vis.poll_events()
        vis.update_renderer()
        img = vis.capture_screen_float_buffer(True)
        vis.destroy_window()
        if len(seg_list) == 1:
            ax.imshow(np.asarray(img))
        else:
            ax[idx].imshow(np.asarray(img))
    plt.savefig(savepath)

def axis2lineset(axis, q=np.array([0.7, 0.0, 0.72]), use_q=False):
    if use_q:
        start_point = q
    else:
        start_point = np.array([0.7, 0.0, 0.72])
    direction_vector = axis[:3]
    direction_vector /= np.linalg.norm(direction_vector)
    line_length = 1
    end_point = start_point + line_length * direction_vector
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.vstack([start_point, end_point]))
    line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
    return line_set

def visualize_pcl_axis(axis_list, NUM_POINT, points, savepath, use_q=False):
    fig, ax = plt.subplots(1,len(axis_list))
    for idx, axis in enumerate(axis_list):
        # setup
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        # viz pcl
        org_colors = np.tile([0.0, 0.0, 1.0], (NUM_POINT, 1)) # blue
        o3d_pcl = o3d.geometry.PointCloud()
        o3d_pcl.points = o3d.utility.Vector3dVector(points)
        o3d_pcl.colors = o3d.utility.Vector3dVector(org_colors)
        vis.add_geometry(o3d_pcl)
        vis.update_geometry(o3d_pcl)
        # viz screw axis
        line_set = axis2lineset(axis, use_q)
        vis.add_geometry(line_set)
        vis.update_geometry(line_set)
        # render
        vis.poll_events()
        vis.update_renderer()
        img = vis.capture_screen_float_buffer(True)
        vis.destroy_window()
        if len(axis_list) == 1:
            ax.imshow(np.asarray(img))
        else:
            ax[idx].imshow(np.asarray(img))
    plt.savefig(savepath)