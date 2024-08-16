import sys, glob, yaml
sys.path.append('../toolbox/')
sys.path.append('')
from lambda_calc import *
from multi_robot import *
from dx200_motion_program_exec_client import *
from WeldSend import *
from RobotRaconteur.Client import *
from weldRRSensor import *
from motoman_def import *
from flir_toolbox import *
import weld_dh2v
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.optimize import Bounds 
from numpy.linalg import norm

def v_opt(v_next, h_err, h_targ, lam=None):
    return norm(h_targ+h_err-weld_dh2v.v2dh_loglog(v_next), 2)**2

bounds = Bounds(3, 17)

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy

def avg_by_line(job_line, flame_pos, num_segs):
    ref_idx = job_line[0]
    job_line_unique = [ref_idx]
    idx = 0
    num_points = 0
    average_pos = []
    sum = np.zeros(flame_pos.shape[1])
    while True:
        while job_line[idx] == ref_idx:
            sum = sum + flame_pos[idx,:]
            num_points += 1
            idx+=1
            if idx >= len(job_line): break
        
        average_pos.append(sum/num_points)
        sum = np.zeros(flame_pos.shape[1])
        try:
            ref_idx = job_line[idx]
            job_line_unique.append(ref_idx)
        except:
            break
        num_points = 0
    average_pos = np.array(average_pos)
    output = np.empty((num_segs, 3))
    
    # for i, line_no in enumerate(job_line_unique):
    #     output[line_no,1:] = average_pos[i,:]
    #     output[line_no,0] = line_no
    for i in range(num_segs):
         if i in job_line_unique:
              idx = job_line_unique.index(i)
              output[i,:] = average_pos[idx, :]
         else:
              output[i,:] = [None, None, None]

    return output

def interpolate_heights(sparse_height):
	# for i, height in enumerate(sparse_height):
	# 	if np.greater(height, 200).any(): sparse_height[i,:] = [None, None, None]
	for i, height in enumerate(sparse_height):
		if np.isnan(height):
			prev_height = sparse_height[i-1]
			#search for next nan number
			for j, height_check in enumerate(sparse_height[i:]):
				if not np.isnan(height_check):
					idx_dif = j+1
					height_dif = (prev_height-height_check)
					for k in range(idx_dif):
						sparse_height[i+k] = prev_height - height_dif*(k+1)/idx_dif
					break
				elif i+j == len(sparse_height)-1:
					idx_dif = j+1
					height_dif = (prev_height-0)
					for k in range(idx_dif):
						sparse_height[i+k] = prev_height - height_dif*(k+1)/idx_dif
	return sparse_height

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

##############################################################SENSORS####################################################################
# weld state logging
# weld_ser = RRN.SubscribeService('rr+tcp://192.168.55.10:60823?service=welder')
cam_ser= None # RRN.ConnectService('rr+tcp://localhost:60827/?service=camera')
# mic_ser = RRN.ConnectService('rr+tcp://192.168.55.20:60828?service=microphone')
## RR sensor objects
rr_sensors = WeldRRSensor(weld_service=None,cam_service=cam_ser,microphone_service=None)

config_dir='../config/'
flir_intrinsic=yaml.load(open(config_dir+'FLIR_A320.yaml'), Loader=yaml.FullLoader)

################################ Data Directories ###########################
now = datetime.now()

dataset='bent_tube/'
sliced_alg='slice_ER_4043/'
data_dir='../data/'+dataset+sliced_alg
rec_folder = input("Enter folder of desired test directory (leave blank for new): ")
if rec_folder == '':
	recorded_dir=now.strftime('../../recorded_data/ER4043_bent_tube_%Y_%m_%d_%H_%M_%S/')
else:
	recorded_dir='../../recorded_data/'+rec_folder+'/'
with open(data_dir+'slicing.yml', 'r') as file:
	slicing_meta = yaml.safe_load(file)
# recorded_dir='recorded_data/cup_ER316L/'
waypoint_distance=1.5
layer_width_num=int(3/slicing_meta['line_resolution'])


robot=robot_obj('MA2010_A0',def_path='../config/MA2010_A0_robot_default_config.yml',tool_file_path='../config/torch.csv',\
	pulse2deg_file_path='../config/MA2010_A0_pulse2deg_real.csv',d=15)
robot2=robot_obj('MA1440_A0',def_path='../config/MA1440_A0_robot_default_config.yml',tool_file_path='../config/flir.csv',\
	pulse2deg_file_path='../config/MA1440_A0_pulse2deg_real.csv',base_transformation_file='../config/MA1440_pose.csv')
positioner=positioner_obj('D500B',def_path='../config/D500B_robot_extended_config.yml',tool_file_path='../config/positioner_tcp.csv',\
	pulse2deg_file_path='../config/D500B_pulse2deg_real.csv',base_transformation_file='../config/D500B_pose.csv')

H2010_1440=H_inv(robot2.base_H)

client=MotionProgramExecClient()
ws=WeldSend(client)

H = np.loadtxt(data_dir+'curve_pose.csv', delimiter=',')
p = H[:3, -1]
R = H[:3, :3]

###set up control parameters
job_offset=200 		###200 for Aluminum ER4043, 300 for Steel Alloy ER70S-6, 400 for Stainless Steel ER316L
nominal_feedrate=160
nominal_vd_relative=9
nominal_wire_length=25 #pixels
nominal_temp_below=500
base_feedrate_cmd=300
base_vd=5
feedrate_cmd=nominal_feedrate
vd_relative=nominal_vd_relative
feedrate_gain=0.5
feedrate_min=100
feedrate_max=300
nominal_slice_increment=1#int(1.05/slicing_meta['line_resolution'])
slice_inc_gain=3.
vd_max=10
feedrate_cmd_adjustment=0
vd_relative_adjustment=0
measure_distance = 500

# ###set up control parameters
# job_offset=300 		###200 for Aluminum ER4043, 300 for Steel Alloy ER70S-6, 400 for Stainless Steel ER316L
# nominal_feedrate=100
# nominal_vd_relative=8
# nominal_wire_length=25 #pixels
# nominal_temp_below=500
# base_feedrate_cmd=300
# base_vd=5
# feedrate_cmd=nominal_feedrate
# vd_relative=nominal_vd_relative
# feedrate_gain=0.5
# feedrate_min=80
# feedrate_max=300
# nominal_slice_increment=1#int(0.85/slicing_meta['line_resolution'])
# slice_inc_gain=3.
# vd_max=10
# feedrate_cmd_adjustment=0
# vd_relative_adjustment=0

# ###set up control parameters
# job_offset=400 		###200 for Aluminum ER4043, 300 for Steel Alloy ER70S-6, 400 for Stainless Steel ER316L
# nominal_feedrate=130
# nominal_vd_relative=8
# nominal_wire_length=25 #pixels
# nominal_temp_below=500
# base_feedrate_cmd=300
# base_vd=5
# feedrate_cmd=nominal_feedrate
# vd_relative=nominal_vd_relative
# feedrate_gain=0.5
# feedrate_min=80
# feedrate_max=300
# #not sure if this nominal slice increment setting will work
# nominal_slice_increment= 1 # int(1/slicing_meta['line_resolution']) # changed this based on mean in wall_gen
# slice_inc_gain=3.
# vd_max=10
# feedrate_cmd_adjustment=0
# vd_relative_adjustment=0


###########################################BASE layer welding############################################
# num_layer_start=int(0*nominal_slice_increment)	###modify layer num here
# num_layer_end=int(1*nominal_slice_increment)
# #q_prev=client.getJointAnglesDB(positioner.pulse2deg)
# q_prev=np.array([9.53E-02,-2.71E+00])	###for motosim tests only

# for layer in range(num_layer_start,num_layer_end,nominal_slice_increment):
# 	mp=MotionProgram(ROBOT_CHOICE='RB1',ROBOT_CHOICE2='ST1',pulse2deg=robot.pulse2deg,pulse2deg_2=positioner.pulse2deg, tool_num = 12)

# 	num_sections=len(glob.glob(data_dir+'curve_sliced_relative/slice'+str(layer)+'_*.csv'))

# 	####################DETERMINE CURVE ORDER##############################################
# 	for x in range(0,num_sections,layer_width_num):
# 		curve_sliced_js=np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(layer)+'_'+str(x)+'.csv',delimiter=',').reshape((-1,6))
# 		if len(curve_sliced_js)<2:
# 			continue
# 		positioner_js=np.loadtxt(data_dir+'curve_sliced_js/D500B_js'+str(layer)+'_'+str(x)+'.csv',delimiter=',')
# 		curve_sliced_relative=np.loadtxt(data_dir+'curve_sliced_relative/slice'+str(layer)+'_'+str(x)+'.csv',delimiter=',')

# 		lam1=calc_lam_js(curve_sliced_js,robot)
# 		lam2=calc_lam_js(positioner_js,positioner)
# 		lam_relative=calc_lam_cs(curve_sliced_relative)

# 		num_points_layer=max(2,int(lam_relative[-1]/waypoint_distance))

# 		###find which end to start depending on how close to joint limit
# 		if positioner.upper_limit[1]-q_prev[1]>q_prev[1]-positioner.lower_limit[1]:
# 			breakpoints=np.linspace(0,len(curve_sliced_js)-1,num=num_points_layer).astype(int)
# 		else:
# 			breakpoints=np.linspace(len(curve_sliced_js)-1,0,num=num_points_layer).astype(int)

# 		#s1_all,s2_all=calc_individual_speed(base_vd,lam1,lam2,lam_relative,breakpoints)

# 		q1_all=[curve_sliced_js[breakpoints[0]]]
# 		q2_all=[positioner_js[breakpoints[0]]]
# 		v1_all=[1]
# 		v2_all=[10]
# 		primitives=['movej']
# 		for j in range(1,len(breakpoints)):
# 			q1_all.append(curve_sliced_js[breakpoints[j]])
# 			q2_all.append(positioner_js[breakpoints[j]])
# 			v1_all.append(max(base_vd,0.1))
# 			positioner_w=vd_relative/np.linalg.norm(curve_sliced_relative[breakpoints[j]][:2])
# 			v2_all.append(min(100,100*positioner_w/positioner.joint_vel_limit[1]))
# 			primitives.append('movel')

# 		q_prev=positioner_js[breakpoints[-1]]
# 		global_ts, timestamp_robot,joint_recording,job_line,_=ws.weld_segment_dual(primitives,robot,
# 																			 positioner,q1_all,q2_all,v1_all,v2_all,cond_all=[int(base_feedrate_cmd/10)+job_offset],arc=False)
# 		q_0 = client.getJointAnglesMH(robot.pulse2deg)[0]
# 		ws.jog_single(robot,[q_0,0,0,0,0,0],4)
# 		input("-------Base Layer Finished-------")






###########################################layer welding############################################
print('----------Normal Layers-----------')
num_layer_start=int(1*nominal_slice_increment)	###modify layer num here
num_layer_end=min(80*nominal_slice_increment,slicing_meta['num_layers'])

#q_prev=client.getJointAnglesDB(positioner.pulse2deg)
q_prev=np.array([9.53E-02,-2.71E+00])	###for motosim tests only
num_sections_prev=5
if num_layer_start<=1*nominal_slice_increment:
	num_sections=len(glob.glob(data_dir+'curve_sliced_relative/slice0_*.csv'))
else:
	num_sections=1


print("start layer: ", num_layer_start)
print("end layer: ", num_layer_end)
print("nominal_slice_increment", nominal_slice_increment)
layer_angle = np.array((slicing_meta['layer_angle']))
base_thickness = slicing_meta['baselayer_thickness']

start_dir = True # Alternate direction that the layer starts from

# for layer in range(num_layer_start,num_layer_end,nominal_slice_increment):
for layer in range(num_layer_start,num_layer_end,nominal_slice_increment):
	if layer != 1:
		save_path = recorded_dir+'layer_'+str(layer-1)+'/'
		start_dir_prev = np.loadtxt(save_path+'start_dir.csv', delimiter=',')
		start_dir = not start_dir_prev
	mp=MotionProgram(ROBOT_CHOICE='RB1',ROBOT_CHOICE2='ST1',pulse2deg=robot.pulse2deg,pulse2deg_2=positioner.pulse2deg, tool_num = 12)

	num_sections_prev=num_sections
	num_sections=len(glob.glob(data_dir+'curve_sliced_relative/slice'+str(layer)+'_*.csv'))

	####################DETERMINE CURVE ORDER##############################################
	x=0
	curve_sliced_js=np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(layer)+'_'+str(x)+'.csv',delimiter=',').reshape((-1,6))
	if len(curve_sliced_js)<2:
		continue
	positioner_js=np.loadtxt(data_dir+'curve_sliced_js/D500B_js'+str(layer)+'_'+str(x)+'.csv',delimiter=',')
	curve_sliced_relative=np.loadtxt(data_dir+'curve_sliced_relative/slice'+str(layer)+'_'+str(x)+'.csv',delimiter=',')
	curve_sliced = np.loadtxt(data_dir+'curve_sliced/slice'+str(layer)+'_'+str(x)+'.csv',delimiter=',')

	###alternate the start point on different ends
	num_points_layer = len(curve_sliced_js)
	print(num_points_layer)
	if start_dir: breakpoints=np.linspace(0,len(curve_sliced_js)-1,num=num_points_layer).astype(int)
	else:
		breakpoints=np.linspace(len(curve_sliced_js)-1,0,num=num_points_layer).astype(int)

	# jog to start and position camera
	# p_positioner_home=np.mean([robot.fwd(curve_sliced_js[0]).p,robot.fwd(curve_sliced_js[-1]).p],axis=0)
	# p_robot2_proj=p_positioner_home+np.array([0,0,50])
	# p2_in_base_frame=np.dot(H2010_1440[:3,:3],p_robot2_proj)+H2010_1440[:3,3]
	# v_z=H2010_1440[:3,:3]@np.array([0,-0.96592582628,-0.2588190451]) ###pointing toward positioner's X with 15deg tiltd angle looking down
	# v_y=VectorPlaneProjection(np.array([-1,0,0]),v_z)	###FLIR's Y pointing toward 1440's -X in 1440's base frame, projected on v_z's plane
	# v_x=np.cross(v_y,v_z)
	# p2_in_base_frame=p2_in_base_frame-measure_distance*v_z			###back project measure_distance-mm away from torch
	# R2=np.vstack((v_x,v_y,v_z)).T
	# q2=robot2.inv(p2_in_base_frame,R2,last_joints=np.zeros(6))[0]
	# q_prev=client.getJointAnglesDB(positioner.pulse2deg)
	# num2p=np.round((q_prev-positioner_js[0])/(2*np.pi))
	# positioner_js+=num2p*2*np.pi
	# ws.jog_dual(robot2,positioner,q2,positioner_js[0],v=1)

	###########################################velocity profile#########################################
	dh_max = slicing_meta['dh_max']
	dh_min = slicing_meta['dh_min']
	point_of_rotation = np.array((slicing_meta['point_of_rotation'],slicing_meta['baselayer_thickness']))
	##calculate distance to point of rotation
	dist_to_por = []
	for i in range(len(curve_sliced)):
		point = np.array((curve_sliced[i,0],curve_sliced[i,2]))
		dist = np.linalg.norm(point-point_of_rotation)
		dist_to_por.append(dist)
	
	
	
	height_profile = [] 
	for distance in dist_to_por:height_profile.append(distance*np.sin(np.deg2rad(layer_angle)))
	vel_nom = weld_dh2v.dh2v_loglog(height_profile, feedrate_cmd, 'ER_4043')

	################################# Check for correction data, and plan velocity accordingly #############################
	if layer == 1:
		velocity_profile = vel_nom
	# else: 
	# 	save_path = recorded_dir+'layer_'+str(layer-1)+'/'
		
	# 	# Process IR data
	# 	with open(save_path+'ir_recording.pickle', 'rb') as file:
	# 		ir_recording = pickle.load(file)
	# 	ir_ts=np.loadtxt(save_path+'ir_stamps.csv', delimiter=',')
	# 	joint_angle=np.loadtxt(save_path+'weld_js_exe.csv', delimiter=',')
	# 	if len(ir_ts) == 0:
	# 		print("No Flame Detected, nominal velocity")
	# 		velocity_profile = vel_nom
	# 	else:
	# 		timeslot=[ir_ts[0]-ir_ts[0], ir_ts[-1]-ir_ts[0]]
	# 		duration=np.mean(np.diff(timeslot))

	# 		flame_3d=[]
	# 		job_no = []
	# 		torch_path = []
	# 		for start_time in timeslot[:-1]:
				
	# 			start_idx=np.argmin(np.abs(ir_ts-ir_ts[0]-start_time))
	# 			end_idx=np.argmin(np.abs(ir_ts-ir_ts[0]-start_time-duration))
			
	# 			#find all pixel regions to record from flame detection
	# 			for i in range(start_idx,end_idx):
					
	# 				ir_image = ir_recording[i]
	# 				try:
	# 					centroid, bbox=flame_detection_aluminum(ir_image, percentage_threshold=0.8)
	# 				except ValueError:
	# 					centroid = None
	# 				if centroid is not None:
	# 					#find spatial vector ray from camera sensor
	# 					vector=np.array([(centroid[0]-flir_intrinsic['c0'])/flir_intrinsic['fsx'],(centroid[1]-flir_intrinsic['r0'])/flir_intrinsic['fsy'],1])
	# 					vector=vector/np.linalg.norm(vector)
	# 					#find index closest in time of joint_angle
	# 					joint_idx=np.argmin(np.abs(ir_ts[i]-joint_angle[:,0]))
	# 					robot2_pose_world=robot2.fwd(joint_angle[joint_idx][8:-2],world=True)
	# 					p2=robot2_pose_world.p
	# 					v2=robot2_pose_world.R@vector
	# 					robot1_pose=robot.fwd(joint_angle[joint_idx][2:8])
	# 					p1=robot1_pose.p
	# 					v1=robot1_pose.R[:,2]
	# 					positioner_pose=positioner.fwd(joint_angle[joint_idx][-2:], world=True)

	# 					#find intersection point
	# 					intersection=line_intersect(p1,v1,p2,v2)
	# 					intersection = positioner_pose.R.T@(intersection-positioner_pose.p)
	# 					torch = positioner_pose.R.T@(robot1_pose.p-positioner_pose.p)

	# 					flame_3d.append(intersection)
	# 					torch_path.append(intersection)
	# 					job_no.append(int(joint_angle[joint_idx][1]))


	# 		###################### Plot Flame vs Anticipated Layers ############################
	# 		print("Flame Processed | Plotting Now")

	# 		flame_3d=np.array(flame_3d)
	# 		torch_path = np.array(torch_path)
	# 		print(flame_3d.shape)
			
	# 		#plot the flame 3d
	# 		fig = plt.figure()
	# 		ax = fig.add_subplot(111, projection='3d')
	# 		try:
	# 			ax.scatter(flame_3d[:,0],flame_3d[:,1],flame_3d[:,2], 'b')
	# 			ax.plot3D(torch_path[:,0], torch_path[:,1], torch_path[:,2])
	# 		except IndexError:
	# 			print("No Flame Detected")
			
	# 		curve_sliced_relative=np.loadtxt(data_dir+'curve_sliced_relative/slice'+str(layer)+'_'+str(x)+'.csv',delimiter=',')
	# 		ax.plot3D(curve_sliced_relative[:,0], curve_sliced_relative[:,1], curve_sliced_relative[:,2])
	# 		ax.set_aspect('equal')
	# 		plt.show()

	# 		#################### Get Height Profile ######################
	# 		for i in range(flame_3d.shape[0]):
	# 			flame_3d[i] = R.T@flame_3d[i]
	# 		print(flame_3d[:,2])
	# 		print(np.any(flame_3d[:,2]).any()>400)
	# 		if (np.greater(flame_3d, 400).any()):
	# 			print("Flame out of bounds Error")
	# 			velocity_profile = vel_nom
	# 		else:
	# 			to_flat_angle = np.deg2rad(layer_angle*(layer-1))
	# 			new_x, new_z = rotate(point_of_rotation, (flame_3d[:,0], flame_3d[:,2]), to_flat_angle)
	# 			flame_3d[:,0] = new_x
	# 			flame_3d[:,2] = new_z-base_thickness

	# 			job_no_offset = 4 # get rid of first non-instructive job lines in AAA file
	# 			job_no = [i-4 for i in job_no]
	# 			averages = avg_by_line(job_no, flame_3d, len(breakpoints))
	# 			scan_height = interpolate_heights(averages[:,-1]) # checks for empty height quantities and fills them in

	# 			height_err = 0-scan_height
	# 			opt_result=minimize(v_opt, vel_nom, (height_err, height_profile), bounds=bounds,
	# 						options = {'maxfun':100000})
	# 			if not opt_result.success: 
	# 				print(opt_result)
	# 				raise ValueError(opt_result.message)
				
	# 			velocity_profile = opt_result.x
	# 			if start_dir_prev == False:
	# 				velocity_profile = np.flip(velocity_profile)
	# 		###########################################################################
	
	
	
	
	
	print(velocity_profile[breakpoints])
	print("Start dir: ", start_dir)
	if layer != 1: print("Prev start dir: ", start_dir_prev)
	input("Check Vel Profile, enter to continue")
	print("Breakpoints Length: ", len(breakpoints))

	save_path = recorded_dir+'layer_'+str(layer)+'/'
	try:
		os.makedirs(save_path)
	except Exception as e:
		print(e)
	# velocity_profile = np.flip(velocity_profile)
	np.savetxt(save_path + 'velocity_profile.csv', velocity_profile[breakpoints], delimiter=',')
	np.savetxt(save_path + 'start_dir.csv', [start_dir], delimiter=',')
	###move to intermidieate waypoint for collision avoidance if multiple section
	if num_sections!=num_sections_prev:
		waypoint_pose=robot.fwd(curve_sliced_js[breakpoints[0]])
		waypoint_pose.p[-1]+=50
		q1=robot.inv(waypoint_pose.p,waypoint_pose.R,curve_sliced_js[breakpoints[0]])[0]
		q2=positioner_js[breakpoints[0]]
		ws.jog_dual(robot,positioner,q1,q2,v=100)
	q1_all=[curve_sliced_js[breakpoints[0]]]
	q2_all=[positioner_js[breakpoints[0]]]
	v1_all=[4]
	v2_all=[10]
	primitives=['movej']
	for j in range(0,len(breakpoints)):
		q1_all.append(curve_sliced_js[breakpoints[j]])
		q2_all.append(positioner_js[breakpoints[j]])
		v1_all.append(max(velocity_profile[breakpoints[j]],0.1))
		
		positioner_w=vd_relative/np.linalg.norm(curve_sliced_relative[breakpoints[j]][:2])
		v2_all.append(min(100,100*positioner_w/positioner.joint_vel_limit[1]))
		primitives.append('movel')
	q_prev=positioner_js[breakpoints[-1]]

	################ Weld with sensors #############################
	rr_sensors.start_all_sensors()
	global_ts, robot_ts,joint_recording,job_line,_=ws.weld_segment_dual(primitives,robot,positioner,q1_all,q2_all,v1_all,v2_all,cond_all=[int(feedrate_cmd/10)+job_offset],arc=False, blocking=True)
	rr_sensors.stop_all_sensors()
	global_ts = np.reshape(global_ts, (-1,1))
	job_line = np.reshape(job_line, (-1,1))

	print(global_ts.shape)
	print(job_line.shape)
	print(joint_recording.shape)
	# save data
	np.savetxt(save_path+'weld_js_exe.csv', np.hstack((global_ts, job_line, joint_recording)), delimiter=',')
	rr_sensors.save_all_sensors(save_path)

	q_0 = client.getJointAnglesMH(robot.pulse2deg)[0]
	ws.jog_single(robot,[q_0,0,0,0,0,0],4)
	# input("-------Layer Finished-------")

	################# PROCESS IR DATA #############################

	


	
	

	#################### Generate Corrective Action ###################################
	

	
