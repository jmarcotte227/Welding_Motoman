import sys
import matplotlib
sys.path.append('../../toolbox/')
sys.path.append('../scan_tools/')
from robot_def import *
from scan_utils import *
from utils import *
from lambda_calc import *
from general_robotics_toolbox import *
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import time
from copy import deepcopy
import colorsys
import math
import pickle

def pcd2height(scanned_points,z_height_start,bbox_min=(-40,-20,0),bbox_max=(40,20,100),\
                resolution_z=0.1,windows_z=0.2,resolution_x=0.1,windows_x=1,stop_thres=20,\
                stop_thres_w=10,use_points_num=5,width_thres=0.8,Transz0_H=None):

    ##### cross section parameters
    # resolution_z=0.1
    # windows_z=0.2
    # resolution_x=0.1
    # windows_x=1
    # stop_thres=20
    # stop_thres_w=10
    # use_points_num=5 # use the largest/smallest N to compute w
    # width_thres=0.8 # prune width that is too close
    ###################################

    # visualize_pcd([scanned_points])

    ## TODO:align path and scan


    bbox_mesh = o3d.geometry.TriangleMesh.create_box(width=bbox_max[0]-bbox_min[0], height=bbox_max[1]-bbox_min[1], depth=0.1)
    box_move=np.eye(4)
    box_move[0,3]=bbox_min[0]
    box_move[1,3]=bbox_min[1]
    box_move[2,3]=0
    bbox_mesh.transform(box_move)
    
    ##### USE THESE TO VISUALIZE #####
    # visualize_pcd([scanned_points,bbox_mesh])
    # exit()
    ##################################
    ##################### get welding pieces end ########################

    ##### get projection of each z height
    profile_height = {}
    z_max=np.max(np.asarray(scanned_points.points)[:,2])
    for z in np.arange(z_height_start,z_max+resolution_z,resolution_z):
        #### crop z height
        min_bound = (-1e5,-1e5,z-windows_z/2)
        max_bound = (1e5,1e5,z+windows_z/2)
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound,max_bound=max_bound)
        points_proj=scanned_points.crop(bbox)
        ##################
        
        min_bound = bbox_min
        max_bound = bbox_max
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound,max_bound=max_bound)
        welds_points=points_proj.crop(bbox)

        # visualize_pcd([welds_points])

        #### get width with x-direction scanning
        if len(welds_points.points)<stop_thres:
            continue

        profile_p = []
        for x in np.arange(bbox_min[0],bbox_max[0]+resolution_x,resolution_x):
            min_bound = (x-windows_x/2,-1e5,-1e5)
            max_bound = (x+windows_x/2,1e5,1e5)
            bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound,max_bound=max_bound)
            welds_points_x = welds_points.crop(bbox)
            if len(welds_points_x.points)<stop_thres_w:
                continue
            #visualize_pcd([welds_points_x])
            ### get the width
            sort_y=np.argsort(np.asarray(welds_points_x.points)[:,1])
            y_min_index=sort_y[:use_points_num]
            y_max_index=sort_y[-use_points_num:]
            y_mid_index=sort_y[use_points_num:-use_points_num]
            
            ### get y and prune y that is too closed
            y_min_all = np.asarray(welds_points_x.points)[y_min_index,1]
            y_min = np.mean(y_min_all)
            y_max_all = np.asarray(welds_points_x.points)[y_max_index,1]
            y_max = np.mean(y_max_all)

            actual_y_min_all=[]
            actual_y_max_all=[]
            for num_i in range(use_points_num):
                if (y_max-y_min_all[num_i])>width_thres:
                    actual_y_min_all.append(y_min_all[num_i])
                if (y_max_all[num_i]-y_min)>width_thres:
                    actual_y_max_all.append(y_max_all[num_i])
            #########
            y_max=0
            y_min=0
            if len(actual_y_max_all)!=0 and len(actual_y_min_all)!=0:
                y_max=np.mean(actual_y_max_all)
                y_min=np.mean(actual_y_min_all)

            this_width=y_max-y_min
            # z_height_ave = np.mean(np.asarray(welds_points_x.points)[np.append(y_min_index,y_max_index),2])
            # z_height_ave = np.mean(np.asarray(welds_points_x.points)[:,2])
            z_height_max = np.max(np.asarray(welds_points_x.points)[:,2])
            profile_p.append(np.array([x,this_width,z_height_max]))
        profile_p = np.array(profile_p)
        
        for pf_i in range(len(profile_p)):
            profile_height[profile_p[pf_i][0]] = profile_p[pf_i][2]

    profile_height_arr = []
    for x in profile_height.keys():
        profile_height_arr.append(np.array([x,profile_height[x]]))
    profile_height_arr=np.array(profile_height_arr)

    profile_height_arr_argsort = np.argsort(profile_height_arr[:,0])
    profile_height_arr=profile_height_arr[profile_height_arr_argsort]
    
    return profile_height_arr,Transz0_H


if __name__ == "__main__":
    data_dir=''
    config_dir='../../config/'

    ######## read the combined point clouds
    scanned_points_mesh = o3d.io.read_triangle_mesh(data_dir+'bent_wall_scan_trimmed.stl')

    scanned_points = scanned_points_mesh.sample_points_uniformly(number_of_points=1110000)

    plane_model, inliers = scanned_points.segment_plane(distance_threshold=float(0.75),
                                                ransac_n=int(5),
                                                num_iterations=int(3000))
    ## Transform the plane to z=0
    plain_norm = plane_model[:3]/np.linalg.norm(plane_model[:3])
    k = np.cross(plain_norm,[0,0,1])
    k = k/np.linalg.norm(k)
    theta = np.arccos(plain_norm[2])
    Transz0 = Transform(rot(k,theta),[0,0,0])*\
                Transform(np.eye(3),[0,0,plane_model[3]/plane_model[2]])
    Transz0_H=H_from_RT(Transz0.R,Transz0.p)
    scanned_points.transform(Transz0_H)

        ## Transform such that the path is in x-axis
    rotation_theta=np.radians(-27) ## rotation angle such that path align x-axis
    translation_p = np.array([-20,80,0]) ## CHANGE THIS TO CALIBRATE NEW SAMPLE


    Trans_zaxis=np.eye(4)
    Trans_zaxis[:3,:3]=rot([0,0,1],rotation_theta)
    Trans_zaxis[:3,3]=translation_p
    scanned_points.transform(Trans_zaxis)
    scanned_points.paint_uniform_color([0.5, 0.5, 0.5])
    visualize_pcd([scanned_points])
    exit()

    box_width = 130
    box_height = 40
    x_offset = 0
    y_offset = -box_height/2

    row_offset = 40
    col_offset = 15

    all_heights = []

    for j in range(1): #row index
        for i in range(1): #col index
            bbox_min=(x_offset-row_offset*i,y_offset-col_offset*j,0)
            bbox_max=(x_offset+box_width-row_offset*i,y_offset+box_height-col_offset*j,100)
            print(bbox_max)
            heights,_ = pcd2height(scanned_points, 1, bbox_min, bbox_max)
            heights[:,0] = heights[:,0]+row_offset*i
            all_heights.append(heights)
    for height in all_heights:
        plt.plot(height[:,0],height[:,1])
    plt.show()
    
    #pickle.dump(all_heights, open(data_dir+'gom_data.pickle','wb'))
