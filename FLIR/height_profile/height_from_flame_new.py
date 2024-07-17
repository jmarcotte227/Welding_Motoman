import yaml
import pickle, sys, os, inspect
import numpy as np
import matplotlib.pyplot as plt
from flir_toolbox import *
from motoman_def import *
import torch_tracking 
from ultralytics import YOLO

def flame_detection_aluminum(raw_img,threshold=1.0e4,area_threshold=4,percentage_threshold=0.8):
    ###flame detection by raw counts thresholding and connected components labeling
    #centroids: x,y
    #bbox: x,y,w,h
    ###adaptively increase the threshold to 60% of the maximum pixel value
    threshold=max(threshold,percentage_threshold*np.max(raw_img))
    thresholded_img=(raw_img>threshold).astype(np.uint8)

    nb_components, labels, stats, centroids = cv2.connectedComponentsWithStats(thresholded_img, connectivity=4)
    
    valid_indices=np.where(stats[:, cv2.CC_STAT_AREA] > area_threshold)[0][1:]  ###threshold connected area
    if len(valid_indices)==0:
        return None, None, None, None
    
    average_pixel_values = [np.mean(raw_img[labels == label]) for label in valid_indices]   ###sorting
    valid_index=valid_indices[np.argmax(average_pixel_values)]      ###get the area with largest average brightness value

    # Extract the centroid and bounding box of the largest component
    centroid = centroids[valid_index]
    bbox = stats[valid_index, :-1]

    return centroid, bbox

config_dir='../../config/'

dataset='bent_tube/'
sliced_alg='slice_ER_4043_dense/'
path_dir='../../data/'+dataset+sliced_alg

# Load Anticipated path
curve_sliced_relative=np.loadtxt(path_dir+'curve_sliced_relative/slice1_0.csv',delimiter=',')

robot=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'torch.csv',\
	pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',d=15)
robot2=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'flir.csv',\
	pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv',base_transformation_file=config_dir+'MA1440_pose.csv')
positioner=positioner_obj('D500B',def_path=config_dir+'D500B_robot_default_config.yml',tool_file_path=config_dir+'positioner_tcp.csv',\
	pulse2deg_file_path=config_dir+'D500B_pulse2deg_real.csv',base_transformation_file=config_dir+'D500B_pose_mocap.csv')

flir_intrinsic=yaml.load(open(config_dir+'FLIR_A320.yaml'), Loader=yaml.FullLoader)
yolo_model = YOLO(os.path.dirname(inspect.getfile(torch_tracking))+"/torch.pt")

# Load the IR recording data from the pickle file
# data_dir='../../../recorded_data/wallbf_100ipm_v10_100ipm_v10/'
data_dir='../../../recorded_data/ER4043_bent_tube/'
with open(data_dir+'/ir_recording.pickle', 'rb') as file:
    ir_recording = pickle.load(file)
ir_ts=np.loadtxt(data_dir+'/ir_stamps.csv', delimiter=',')
joint_angle=np.loadtxt(data_dir+'/weld_js_exe.csv', delimiter=',')

timeslot=[ir_ts[0]-ir_ts[0], ir_ts[-1]-ir_ts[0]]#[124.7,135.1,145.6,156.0,166.5,176.9,187.8,198.3,208.9,219.2,229.8,240.3,250.8,261.2,271.8,282.2,292.7,303.2,313.7,324.2,334.7,345.3,355.8,366.3]
# timeslot=[0.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800]
duration=np.mean(np.diff(timeslot))

print("Number of frames:", ir_recording.shape)
print("Number of timestesps: ", ir_ts.shape)
print("duration: ", duration)



flame_3d=[]
for start_time in timeslot[:-1]:
    
    start_idx=np.argmin(np.abs(ir_ts-ir_ts[0]-start_time))
    end_idx=np.argmin(np.abs(ir_ts-ir_ts[0]-start_time-duration))
    print(start_idx)
    print(end_idx)
   
    #find all pixel regions to record from flame detection
    for i in range(start_idx,end_idx):
        
        ir_image = ir_recording[i]
        try:
            centroid, bbox=flame_detection_aluminum(ir_image, percentage_threshold=0.8)
        except ValueError:
            centroid = None
        if centroid is not None:
            #find spatial vector ray from camera sensor
            vector=np.array([(centroid[0]-flir_intrinsic['c0'])/flir_intrinsic['fsx'],(centroid[1]-flir_intrinsic['r0'])/flir_intrinsic['fsy'],1])
            vector=vector/np.linalg.norm(vector)
            #find index closest in time of joint_angle
            joint_idx=np.argmin(np.abs(ir_ts[i]-joint_angle[:,0]))
            robot2_pose_world=robot2.fwd(joint_angle[joint_idx][8:-2],world=True)
            p2=robot2_pose_world.p
            v2=robot2_pose_world.R@vector
            robot1_pose=robot.fwd(joint_angle[joint_idx][2:8])
            p1=robot1_pose.p
            v1=robot1_pose.R[:,2]
            positioner_pose=positioner.fwd(joint_angle[joint_idx][-2:], world=True)

            #find intersection point
            intersection=line_intersect(p1,v1,p2,v2)
            intersection = positioner_pose.R@(intersection-positioner_pose.p)

            flame_3d.append(intersection)

            ##########################################################DEBUGGING & VISUALIZATION: plot out p1,v1,p2,v2,intersection##########################################################
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax.scatter(p1[0],p1[1],p1[2],c='r',label='robot1')
            # ax.quiver(p1[0],p1[1],p1[2],v1[0],v1[1],v1[2],color='r',label='robot1_ray',length=100)
            # ax.scatter(p2[0],p2[1],p2[2],c='b',label='robot2')
            # ax.quiver(p2[0],p2[1],p2[2],v2[0],v2[1],v2[2],color='b',label='robot2_ray',length=100)
            # ax.quiver(p2[0],p2[1],p2[2],robot2_pose_world.R[0,2],robot2_pose_world.R[1,2],robot2_pose_world.R[2,2],color='g',label='optical_axis',length=100)
            # ax.scatter(intersection[0],intersection[1],intersection[2],c='g',label='intersection')

            # ax.legend()
            # ax.set_xlabel('X')
            # ax.set_ylabel('Y')
            # ax.set_zlabel('Z')
            # plt.show()

# print("Flame Processed | Plotting Now")
# flame_3d=np.array(flame_3d)
# #plot the flame 3d
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(flame_3d[:,0],flame_3d[:,1],flame_3d[:,2], c='r')
# ax.plot3D(curve_sliced_relative[:,0], curve_sliced_relative[:,1], curve_sliced_relative[:,2], c='b')
# #set equal aspect ratio
# ax.set_box_aspect([np.ptp(flame_3d[:,0]),np.ptp(flame_3d[:,1]),np.ptp(flame_3d[:,2])])
# ax.set_aspect('equal')
# plt.show()

print("Flame Processed | Plotting Now")

flame_3d=np.array(flame_3d)
print(flame_3d.shape)
#plot the flame 3d
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(flame_3d[:700,0],flame_3d[:700,1],flame_3d[:700,2], 'b')
# plot slice and successive slices
layer = 10
x=0
curve_sliced_relative=np.loadtxt(path_dir+'curve_sliced_relative/slice'+str(layer)+'_'+str(x)+'.csv',delimiter=',')
ax.plot3D(curve_sliced_relative[:,0], curve_sliced_relative[:,1], curve_sliced_relative[:,2], c='g')
try:
    for plot_layer in range(layer+2, layer+20, 2):
        curve_sliced_relative=np.loadtxt(path_dir+'curve_sliced_relative/slice'+str(plot_layer)+'_'+str(x)+'.csv',delimiter=',')
        ax.plot3D(curve_sliced_relative[:,0], curve_sliced_relative[:,1], curve_sliced_relative[:,2], c='r') 
        print("Layer above: ", plot_layer) 
except FileNotFoundError:
    print("Layers outside of sliced layers")
try:    
    for plot_layer in range(layer-2, layer-20, -2):
        if plot_layer <=0: raise FileNotFoundError
        curve_sliced_relative=np.loadtxt(path_dir+'curve_sliced_relative/slice'+str(plot_layer)+'_'+str(x)+'.csv',delimiter=',')
        ax.plot3D(curve_sliced_relative[:,0], curve_sliced_relative[:,1], curve_sliced_relative[:,2], c='b')
        print("Layer below: ", plot_layer)
except FileNotFoundError: 
    print("No layers prior")    

#set equal aspect ratio
ax.set_box_aspect([np.ptp(flame_3d[:,0]),np.ptp(flame_3d[:,1]),np.ptp(flame_3d[:,2])])
#ax.set_aspect('equal')
plt.show()