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