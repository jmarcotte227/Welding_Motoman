import sys, copy
import matplotlib

from sklearn import cluster
sys.path.append('../toolbox/')
from pointcloud_toolbox import *
from utils import *
from motoman_def import *
# from scan_utils import *
from general_robotics_toolbox import *
import open3d as o3d

import numpy as np

def visualize_pcd(show_pcd_list,point_show_normal=False):

    show_pcd_list_legacy=[]
    for obj in show_pcd_list:
        if type(obj) is o3d.cpu.pybind.t.geometry.PointCloud or type(obj) is o3d.cpu.pybind.t.geometry.TriangleMesh:
            show_pcd_list_legacy.append(obj.to_legacy())
        else:
            show_pcd_list_legacy.append(obj)

    points_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20,origin=[0,0,0])
    show_pcd_list_legacy.append(points_frame)
    o3d.visualization.draw_geometries(show_pcd_list_legacy,width=960,height=540,point_show_normal=point_show_normal)

    
data_dir='../data/bent_tube/slice_ER_4043/'
scanned_dir='../../recorded_data/ER4043_bent_tube_2024_08_01_11_47_23/'
save_dir = 'saved_pointclouds/'
######## read the scanned stl
# target_mesh = o3d.io.read_triangle_mesh(data_dir+'surface.stl')
scanned_mesh = o3d.io.read_triangle_mesh(scanned_dir+'ER4043_automated_tube_trimmed.stl')
# target_mesh.compute_vertex_normals()
scanned_mesh.compute_vertex_normals()

######## Get Points from all curve_sliced_relative
target_coords = []
for i in range(1,80):
    points = np.loadtxt(data_dir+'curve_sliced_relative/slice'+str(i)+'_0.csv',delimiter=',')
    for row in points:
        target_coords.append(row[:3])
target_coords = np.array(target_coords)
np.savetxt(save_dir+'curve_slice_coords.csv', target_coords, delimiter=',')
    
## inch to mm
# target_mesh.scale(25.4, center=(0, 0, 0))
# visualize_pcd([target_mesh,scanned_mesh])

## sample as pointclouds
# target_points = target_mesh.sample_points_uniformly(number_of_points=111000)
scanned_points = scanned_mesh.sample_points_uniformly(number_of_points=111000)

target_points = o3d.geometry.PointCloud()
target_points.points = o3d.utility.Vector3dVector(target_coords)
## paint colors
target_points = target_points.paint_uniform_color([0, 0.8, 0.0])
scanned_points = scanned_points.paint_uniform_color([0.8, 0, 0.0])
scanned_points_original=copy.deepcopy(scanned_points)
total_transformation = np.array([[ 6.71184034e-01, -7.17519464e-01,  1.86219793e-01,  7.86854660e+01],
                                 [ 7.40874617e-01,  6.57712433e-01, -1.36085110e-01, -1.15101035e+01],
                                 [-2.48353577e-02,  2.29303671e-01,  9.73038042e-01, -2.47471096e+01],
                                 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
angle = 0.7854
total_transformation = total_transformation@np.array([[np.cos(angle), -np.sin(angle), 0, 100],
                                                      [np.sin(angle), np.cos(angle), 0, 75],
                                                      [0, 0, 1, 0],
                                                      [0, 0, 0, 1]])

## global tranformation
# R_guess,p_guess=global_alignment(scanned_points.points,target_points.points)
# total_transformation = H_from_RT(R_guess,p_guess)
scanned_points=scanned_points.transform(total_transformation)


visualize_pcd([target_points,scanned_points])




## ICP
icp_turns = 1
threshold=5
max_iteration=1000
for i in range(icp_turns):
    reg_p2p = o3d.pipelines.registration.registration_icp(
                scanned_points, target_points, threshold, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration))
    scanned_points=scanned_points.transform(reg_p2p.transformation)
    total_transformation = reg_p2p.transformation@total_transformation

print("Final Transformation:",total_transformation)
print("Fitness:",reg_p2p.fitness)
print("inlier_rmse:",reg_p2p.inlier_rmse)
print(reg_p2p)
visualize_pcd([target_points,scanned_points])
