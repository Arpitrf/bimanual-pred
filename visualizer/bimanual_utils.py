"""
Visualization utilities for bimanual prediction
"""
import numpy as np
import open3d as o3d
import matplotlib
import matplotlib.pyplot
matplotlib.use('Agg')
# Monkey-patch torch.utils.tensorboard.SummaryWriter
from open3d.visualization.tensorboard_plugin import summary
# Utility function to convert Open3D geometry to a dictionary format
from open3d.visualization.tensorboard_plugin.util import to_dict_batch


def visualize_pcl_contact(seg_list, NUM_POINT, points, savepath):
    fig, ax = matplotlib.pyplot.subplots(1,len(seg_list))
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
    matplotlib.pyplot.savefig(savepath)

def axis2lineset(s_hat=np.array([0.0, 0.0, 1.0]), q=np.array([0.7, 0.0, 0.72]), use_q=False, use_s=False):
    start_point = q
    # print("start_poitntt: ", start_point)
    direction_vector = s_hat
    direction_vector /= np.linalg.norm(direction_vector)
    line_length = 1
    end_point = start_point + line_length * direction_vector
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.vstack([start_point, end_point]))
    line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
    
    # Create a sphere
    radius = 0.03
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(start_point)

    return line_set, sphere

# def visualize_pcl_axis(axis_list, NUM_POINT, points, savepath, use_q=False):
#     fig, ax = matplotlib.pyplot.subplots(1,len(axis_list))
#     for idx, axis in enumerate(axis_list):
#         # setup
#         vis = o3d.visualization.Visualizer()
#         vis.create_window(visible=False)
#         # viz pcl
#         org_colors = np.tile([0.0, 0.0, 1.0], (NUM_POINT, 1)) # blue
#         o3d_pcl = o3d.geometry.PointCloud()
#         o3d_pcl.points = o3d.utility.Vector3dVector(points)
#         o3d_pcl.colors = o3d.utility.Vector3dVector(org_colors)
#         vis.add_geometry(o3d_pcl)
#         vis.update_geometry(o3d_pcl)
#         # viz screw axis
#         line_set = axis2lineset(axis, use_q)
#         vis.add_geometry(line_set)
#         vis.update_geometry(line_set)
#         # render
#         vis.poll_events()
#         vis.update_renderer()
#         img = vis.capture_screen_float_buffer(True)
#         vis.destroy_window()
#         if len(axis_list) == 1:
#             ax.imshow(np.asarray(img))
#         else:
#             ax[idx].imshow(np.asarray(img))
#     # matplotlib.pyplot.show()
#     matplotlib.pyplot.savefig(savepath)

def visualize_pcl_axis(axis_list, NUM_POINT, points, savepath, use_q=False, use_s=False, writer=None, epoch=None):
    fig, ax = matplotlib.pyplot.subplots(1,len(axis_list))
    pcls = []
    line_sets = []
    spheres = []
    for idx, axis in enumerate(axis_list):
        # setup
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        # ctr = vis.get_view_control()
        # parameters = o3d.io.read_pinhole_camera_parameters("o3d_camera.json")
        # ctr.convert_from_pinhole_camera_parameters(parameters)
        # temp = ctr.convert_to_pinhole_camera_parameters()
        # print("temp: ", temp.intrinsic)

        # -----------
        # ctr = vis.get_view_control()
        # camera_params = ctr.convert_to_pinhole_camera_parameters()
        # K = np.eye(4) 
        # K[:3, :3] = np.array([[0,1,0],
        #                     [1,1,0],
        #                     [0,0,1]])
        # K[:3, -1] = np.array([1,1,3])
        # camera_params.extrinsic = K
        # ctr.convert_from_pinhole_camera_parameters(camera_params)
        # vis.update_renderer()
        # ---------------------------

        # viz pcl
        org_colors = np.tile([0.0, 0.0, 1.0], (NUM_POINT, 1)) # blue
        o3d_pcl = o3d.geometry.PointCloud()
        o3d_pcl.points = o3d.utility.Vector3dVector(points)
        o3d_pcl.colors = o3d.utility.Vector3dVector(org_colors)
        pcls.append(o3d_pcl)
        vis.add_geometry(o3d_pcl)
        vis.update_geometry(o3d_pcl)
        # viz screw axis
        if use_q and use_s:
            line_set, sphere = axis2lineset(s_hat=axis[0], q=axis[1], use_q=use_q, use_s=use_s)
        elif use_q:
            line_set, sphere = axis2lineset(q=axis, use_q=use_q, use_s=use_s)
        else:
            line_set, sphere = axis2lineset(s_hat=axis, use_q=use_q, use_s=use_s)
        # remove later
        # o3d.visualization.draw_geometries([o3d_pcl, line_set, sphere])

        vis.add_geometry(line_set)
        vis.update_geometry(line_set)
        vis.add_geometry(sphere)
        vis.update_geometry(sphere)
        line_sets.append(line_set)
        spheres.append(sphere)
        # render
        vis.poll_events()
        vis.update_renderer()
        img = vis.capture_screen_float_buffer(True)
        vis.destroy_window()
        if len(axis_list) == 1:
            ax.imshow(np.asarray(img))
        else:
            ax[idx].imshow(np.asarray(img))
    # matplotlib.pyplot.show()
    matplotlib.pyplot.savefig(savepath)
    # temp = to_dict_batch(pcls)
    # print("temp.keys(): ", temp.keys())
    # if epoch is not None:
    #     writer.add_3d('pcd', to_dict_batch(pcls + line_sets + spheres), step=epoch)
    # o3d.visualization.draw_geometries(pcls + line_sets + spheres)
