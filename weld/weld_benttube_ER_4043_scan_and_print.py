import sys, glob, yaml
from pathlib import Path
from copy import deepcopy
sys.path.append('../toolbox/')
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
import open3d as o3d

sys.path.append('../toolbox/')
sys.path.append('../scan/scan_tools/')
sys.path.append('../scan/scan_plan/')
sys.path.append('../scan/scan_process/')
sys.path.append('../mocap/')

from scanPathGen import *

def connect_failed(s, client_id, url, err):
    global mti_sub, mti_client
    print ("Client connect failed: " + str(client_id.NodeID) + " url: " + str(url) + " error: " + str(err))
    mti_sub=RRN.SubscribeService(url)
    mti_client=mti_sub.GetDefaultClientWait(1)

def generate_mti_rr():
    
    global mti_sub,mti_client
    
    mti_sub=RRN.SubscribeService("rr+tcp://192.168.55.10:60830/?service=MTI2D")
    mti_sub.ClientConnectFailed += connect_failed
    mti_client=mti_sub.GetDefaultClientWait(1)
    mti_client.setExposureTime("25")


##############################################################SENSORS####################################################################
# weld state logging
weld_ser = RRN.SubscribeService('rr+tcp://192.168.55.10:60823?service=welder')
# cam_ser=RRN.ConnectService('rr+tcp://localhost:60827/?service=camera')
# mic_ser = RRN.ConnectService('rr+tcp://192.168.55.20:60828?service=microphone')
rr_sensors = WeldRRSensor(weld_service=weld_ser)
# scanner init
mti_client = RRN.ConnectService("rr+tcp://192.168.55.10:60830/?service=MTI2D")
mti_client.setExposureTime("25")

# Set up scanner Parameters
scan_speed = 5 # scanning speed (mm/sec)
scan_stand_off_d = 95 # mm
Rz_angle = np.radians(0) # point direction with respect to weld)
Ry_angle = np.radians(0)
bounds_theta = np.radians(1)
all_scan_angle = np.radians([0])
q_init_table = np.radians([-15, 200])

to_home_speed=4
to_start_speed=4

mti_Rpath = np.array([[ -1.,0.,0.],   
                        [ 0.,1.,0.],
                        [0.,0.,-1.]])



## RR sensor objects
rr_sensors = WeldRRSensor(weld_service=None,cam_service=None,microphone_service=None)

################################ Data Directories ###########################
now = datetime.now()

dataset='bent_tube/'
sliced_alg='slice_ER_4043_dense/'
data_dir='../data/'+dataset+sliced_alg
recorded_dir=now.strftime('../../recorded_data/ER4043_bent_tube_%Y_%m_%d_%H_%M_%S/')
with open(data_dir+'slicing.yml', 'r') as file:
	slicing_meta = yaml.safe_load(file)
# recorded_dir='recorded_data/cup_ER316L/'
waypoint_distance=1.5
layer_width_num=int(3/slicing_meta['line_resolution'])


############################# Robot Objects ####################################
R1_ph_dataset_date='0926'
R2_ph_dataset_date='0926'
S1_ph_dataset_date='0926'

zero_config = np.zeros(6)
config_dir='../config/'

R1_marker_dir=config_dir+'MA2010_marker_config/'
weldgun_marker_dir=config_dir+'weldgun_marker_config/'
R2_marker_dir=config_dir+'MA1440_marker_config/'
mti_marker_dir=config_dir+'mti_marker_config/'
S1_marker_dir=config_dir+'D500B_marker_config/'
S1_tcp_marker_dir=config_dir+'positioner_tcp_marker_config/'


robot=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml', tool_file_path=config_dir+'torch.csv',\
	pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',d=15,\
	base_marker_config_file=R1_marker_dir+'MA2010_'+R1_ph_dataset_date+'_marker_config.yaml',\
	tool_marker_config_file=weldgun_marker_dir+'weldgun_'+R1_ph_dataset_date+'_marker_config.yaml')
robot2=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'flir.csv',\
	pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv',base_transformation_file=config_dir+'MA1440_pose.csv',\
	base_marker_config_file=R2_marker_dir+'MA1440_'+R2_ph_dataset_date+'_marker_config.yaml',\
	tool_marker_config_file=mti_marker_dir+'mti_'+R2_ph_dataset_date+'_marker_config.yaml')
positioner=positioner_obj('D500B',def_path=config_dir+'D500B_robot_extended_config.yml',tool_file_path=config_dir+'positioner_tcp.csv',\
	pulse2deg_file_path=config_dir+'D500B_pulse2deg_real.csv',base_transformation_file=config_dir+'D500B_pose.csv')

#### change base H to calibrated ones ####
robot_scan_base = robot.T_base_basemarker.inv()*robot2.T_base_basemarker
robot2.base_H = H_from_RT(robot_scan_base.R,robot_scan_base.p)
positioner_base = robot.T_base_basemarker.inv()*positioner.T_base_basemarker
positioner.base_H = H_from_RT(positioner_base.R,positioner_base.p)
T_to_base = Transform(np.eye(3),[0,0,-380])
positioner.base_H = np.matmul(positioner.base_H,H_from_RT(T_to_base.R,T_to_base.p))

r1_nom_P=np.array([[0,0,0],[150,0,0],[0,0,760],\
                   [1082,0,200],[0,0,0],[0,0,0],[100,0,0]]).T
r1_nom_H=np.array([[0,0,1],[0,1,0],[0,-1,0],\
                [-1,0,0],[0,-1,0],[-1,0,0]]).T
r2_nom_P=np.array([[0,0,0],[155,0,0],[0,0,614],\
                   [640,0,200],[0,0,0],[0,0,0],[100,0,0]]).T
r2_nom_H=np.array([[0,0,1],[0,1,0],[0,-1,0],\
                [-1,0,0],[0,-1,0],[-1,0,0]]).T

R1_mid = np.radians([-25,0,0,0,0,0])
R1_home = np.radians([-60,0,0,0,0,0])
R2_mid = np.radians([-6,20,-10,0,0,0])
R2_home = np.radians([-30,20,-10,0,0,0])

client=MotionProgramExecClient()
ws=WeldSend(client)
scan_process = ScanProcess(robot2, positioner)

# MTI connect to RR
generate_mti_rr()
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




# ###########################################BASE layer welding############################################
# num_layer_start=int(0*nominal_slice_increment)	###modify layer num here
# num_layer_end=int(1*nominal_slice_increment)

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
# 		if start_dir: breakpoints=np.linspace(0,len(curve_sliced_js)-1,num=num_points_layer).astype(int)
#		else:
#			breakpoints=np.linspace(len(curve_sliced_js)-1,0,num=num_points_layer).astype(int)

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

# 		global_ts, timestamp_robot,joint_recording,job_line,_=ws.weld_segment_dual(primitives,robot,positioner,q1_all,q2_all,v1_all,v2_all,cond_all=[int(base_feedrate_cmd/10)+job_offset],arc=True)
# 		q_0 = client.getJointAnglesMH(robot.pulse2deg)[0]
# 		ws.jog_single(robot,[q_0,0,0,0,0,0],4)
# 		input("-------Base Layer Finished-------")






###########################################layer welding############################################
print('----------Normal Layers-----------')
num_layer_start=int(790*nominal_slice_increment)	###modify layer num here
num_layer_end=min(80*nominal_slice_increment,slicing_meta['num_layers'])

num_sections_prev=5
if num_layer_start<=1*nominal_slice_increment:
	num_sections=len(glob.glob(data_dir+'curve_sliced_relative/slice0_*.csv'))
else:
	num_sections=1


print("start layer: ", num_layer_start)
print("end layer: ", num_layer_end)
print("nominal_slice_increment", nominal_slice_increment)

start_dir = False # Alternate direction that the layer starts from
layer = num_layer_start
# for layer in range(num_layer_start,num_layer_end,nominal_slice_increment):
while layer <= int(slicing_meta['num_layers']):
	mp=MotionProgram(ROBOT_CHOICE='RB1',ROBOT_CHOICE2='ST1',pulse2deg=robot.pulse2deg,pulse2deg_2=positioner.pulse2deg, tool_num = 12)

	num_sections_prev=num_sections
	num_sections=len(glob.glob(data_dir+'curve_sliced_relative/slice'+str(layer)+'_*.csv'))

	####################DETERMINE CURVE ORDER##############################################
	

	for x in range(0,num_sections,layer_width_num):
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
		start_dir = not start_dir
		###########################################velocity profile#########################################
		dh_max = slicing_meta['dh_max']
		dh_min = slicing_meta['dh_min']
		point_of_rotation = np.array((slicing_meta['point_of_rotation'],slicing_meta['baselayer_thickness']))
		layer_angle = np.array((slicing_meta['layer_angle']))
		##calculate distance to point of rotation
		dist_to_por = []
		for i in range(len(curve_sliced)):
			point = np.array((curve_sliced[i,0],curve_sliced[i,2]))
			dist = np.linalg.norm(point-point_of_rotation)
			dist_to_por.append(dist)
		
		
		
		height_profile = [] 
		for distance in dist_to_por:height_profile.append(distance*np.sin(layer_angle*np.pi/180))
		velocity_profile = weld_dh2v.dh2v_loglog(height_profile, feedrate_cmd, 'ER_4043')
		print(velocity_profile)
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

		################ Weld with sensors #############################
		rr_sensors.start_all_sensors()
		global_ts, robot_ts,joint_recording,job_line,_=ws.weld_segment_dual(primitives,robot,positioner,q1_all,q2_all,v1_all,v2_all,cond_all=[int(feedrate_cmd/10)+job_offset],arc=True, blocking=True)
		rr_sensors.stop_all_sensors()
		global_ts = np.reshape(global_ts, (-1,1))
		job_line = np.reshape(job_line, (-1,1))

		# save data
		save_path = recorded_dir+'layer_'+str(layer)+'/'
		try:
			os.makedirs(save_path)
		except Exception as e:
			print(e)
		np.savetxt(save_path+'weld_js_exe.csv', np.hstack((global_ts, job_line, joint_recording)), delimiter=',')
		rr_sensors.save_all_sensors(save_path)

		# q_0 = client.getJointAnglesMH(robot.pulse2deg)[0]
		# ws.jog_single(robot,[q_0,0,0,0,0,0],4)
		
	input("-------Layer Finished-------")
	# send R1_home
	ws.jog_single(robot, R1_home, v=4)



	################### scan Part ################################
	scan_st = time.time()
	read_layer = layer


	pcd_layer = o3d.geometry.PointCloud()
	layer_curve_relative = []
	layer_curve_dh = []
	for x in range(0, num_sections):
		spg = ScanPathGen(robot2, positioner, scan_stand_off_d, Rz_angle, Ry_angle, bounds_theta)

		# This might not be necessary, need to check
		# curve_sliced_relative=np.loadtxt(data_dir+'curve_sliced_relative/slice'+str(layer)+'_'+str(x)+'.csv',delimiter=',')
		# curve_sliced_js=np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(layer)+'_'+str(x)+'.csv',delimiter=',').reshape((-1,6))
		# positioner_js=np.loadtxt(data_dir+'curve_sliced_js/D500B_js'+str(layer)+'_'+str(x)+'.csv',delimiter=',')
		# rob_js_plan = np.hstack((curve_sliced_js, positioner_js))

		q_out1=np.loadtxt(data_dir+'curve_scan_js/MA1440_js'+str(read_layer)+'_'+str(x)+'.csv',delimiter=',').reshape((-1,6))
		q_out2=np.loadtxt(data_dir+'curve_scan_js/D500B_js'+str(read_layer)+'_'+str(x)+'.csv',delimiter=',').reshape((-1,2))
		scan_p=np.loadtxt(data_dir+'curve_scan_relative/scan_T'+str(read_layer)+'_'+str(x)+'.csv',delimiter=',')

		lam_relative = calc_lam_cs(scan_p)
		scan_waypoint_distance = waypoint_distance
		num_points_layer=max(2,int(lam_relative[-1]/scan_waypoint_distance))

		breakpoints=np.linspace(0,len(lam_relative)-1,num=num_points_layer).astype(int)
		q_prev=ws.client.getJointAnglesDB(positioner.pulse2deg)
		if np.linalg.norm(q_prev-q_out2[0])>np.linalg.norm(q_prev-q_out2[-1]):
			breakpoints=breakpoints[::-1]
		# generate motion program
		q_bp1,q_bp2,s1_all,s2_all=spg.gen_motion_program(q_out1,q_out2,scan_p,scan_speed,breakpoints=breakpoints,init_sync_move=0)
		v1_all = [1]
		v2_all = [1]
		primitives=['movej']

		for j in range(1,len(breakpoints)):
			v1_all.append(max(s1_all[j-1],0.1))
			positioner_w=scan_speed/np.linalg.norm(scan_p[breakpoints[j]][:2])
			v2_all.append(min(100,100*positioner_w/positioner.joint_vel_limit[1]))
			primitives.append('movel')

		# execute scanning
		input("Scan Move to Start")
		waypoint_pose=robot2.fwd(q_bp1[0][0])
		waypoint_pose.p[-1]+=50

		try:
			q1=robot2.inv(waypoint_pose.p,waypoint_pose.R,q_bp1[0][0])[0]
		except:
			print("Use nom PH for ik")
			robot2.robot.P=deepcopy(r2_nom_P)
			robot2.robot.H=deepcopy(r2_nom_H)
			q1=robot2.inv(waypoint_pose.p,waypoint_pose.R,q_bp1[0][0])[0]
			robot2.robot.P=deepcopy(robot2.calib_P)
			robot2.robot.H=deepcopy(robot2.calib_H)
		
		q2=q_bp2[0][0]

		ws.jog_dual(robot2,positioner,[R2_mid,q1],q2,v=to_start_speed)

		input("Start Scan")
		scan_scan_st=time.time()
		mp = MotionProgram(ROBOT_CHOICE='RB2',ROBOT_CHOICE2='ST1',pulse2deg=robot2.pulse2deg,pulse2deg_2=positioner.pulse2deg)
		target2=['MOVJ',np.degrees(q_bp2[0][0]),to_start_speed]
		mp.MoveJ(np.degrees(q_bp1[0][0]), to_start_speed, 0, target2=target2)
		ws.client.execute_motion_program(mp)

		#constructing actual scan path
		mp = MotionProgram(ROBOT_CHOICE='RB2',ROBOT_CHOICE2='ST1',pulse2deg=robot2.pulse2deg,pulse2deg_2=positioner.pulse2deg)
		for path_i in range(1,len(q_bp1)-1):
			target2=['MOVJ',np.degrees(q_bp2[path_i][0]),v2_all[path_i]]
			mp.MoveL(np.degrees(q_bp1[path_i][0]), v1_all[path_i], target2=target2)
		target2=['MOVJ',np.degrees(q_bp2[-1][0]),v2_all[-1]]
		mp.MoveL(np.degrees(q_bp1[-1][0]), v1_all[-1], 0, target2=target2)

		mti_break_flag=False
		ws.client.execute_motion_program_nonblocking(mp)

		###streaming
		ws.client.StartStreaming()
		start_time=time.time()
		state_flag=0
		joint_recording=[]
		robot_stamps=[]
		mti_recording=None
		mti_recording=[]
		r_pulse2deg = np.append(robot2.pulse2deg,positioner.pulse2deg)
		while True:
			if state_flag & STATUS_RUNNING == 0 and time.time()-start_time>1.:
				break 
			res, fb_data = ws.client.fb.try_receive_state_sync(ws.client.controller_info, 0.001)
			if res:
				joint_angle=np.hstack((fb_data.group_state[0].feedback_position,fb_data.group_state[1].feedback_position,fb_data.group_state[2].feedback_position))
				state_flag=fb_data.controller_flags
				joint_recording.append(joint_angle)
				timestamp=fb_data.time
				robot_stamps.append(timestamp)
				###MTI scans YZ point from tool frame
				try:
					mti_recording.append(deepcopy(np.array([mti_client.lineProfile.X_data,mti_client.lineProfile.Z_data])))
				except Exception as e:
					if not mti_break_flag:
						print(e)
					mti_break_flag=True
		ws.client.servoMH(False)
		print("Actual scan time:",time.time()-scan_scan_st)
		if not mti_break_flag:
			break
		print("MTI broke during robot move")
		while True:
			try:
				input("MTI reconnect ready?")
				generate_mti_rr()
				break
			except:
				pass
		
	
	mti_recording=np.array(mti_recording)
	joint_recording=np.array(joint_recording)
	q_out_exe=joint_recording[:,6:]

	# Saving point data
	layer_data_dir=recorded_dir+'layer_'+str(layer)+'_'+str(x)+'/'
	Path(recorded_dir).mkdir(exist_ok=True)
	Path(layer_data_dir).mkdir(exist_ok=True)
	out_scan_dir = layer_data_dir+'scans/'
	Path(out_scan_dir).mkdir(exist_ok=True)
	## save traj
	# save poses
	np.savetxt(out_scan_dir + 'scan_js_exe.csv',q_out_exe,delimiter=',')
	np.savetxt(out_scan_dir + 'scan_robot_stamps.csv',robot_stamps,delimiter=',')
	with open(out_scan_dir + 'mti_scans.pickle', 'wb') as file:
		pickle.dump(mti_recording, file)
	print('Total scans:',len(mti_recording))

	###################### Process Scans ################################################
	crop_extend=15
	crop_min=tuple(np.min(curve_sliced_relative[:,:3],axis=0)-crop_extend)
	crop_max=tuple(np.max(curve_sliced_relative[:,:3],axis=0)+crop_extend)
	scan_process = ScanProcess(robot2,positioner)
	pcd = scan_process.pcd_register_mti(mti_recording,q_out_exe,robot_stamps,use_calib=True,ph_param=None)

	cluser_minp = 300
	
	pcd_new = scan_process.pcd_noise_remove(pcd,nb_neighbors=40,std_ratio=1.5,\
										min_bound=crop_min,max_bound=crop_max,outlier_remove=True,cluster_based_outlier_remove=True,cluster_neighbor=1,min_points=cluser_minp)
	visualize_pcd([pcd_new]) # Not sure if this is an open 3d function or not
	pcd=pcd_new

	# calibrate H
	pcd,Transz0_H = scan_process.pcd_calib_z(pcd,Transz0_H=Transz0_H)

	###################### Record dh and curve relative #################################
	profile_dh = scan_process.pcd2dh(pcd,curve_sliced_relative, drawing=False)
	if len(layer_curve_dh)!=0:
		profile_dh[:,0]=profile_dh[:,0]+layer_curve_dh[-1][0]
	layer_curve_dh.extend(profile_dh)

	layer_curve_relative.extend(curve_sliced_relative)

	o3d.io.write_point_cloud(out_scan_dir+'processed_pcd.pcd', pcd)
	np.save(out_scan_dir+'height_profile.npy', profile_dh)
	pcd_layer+=pcd

	last_pcd_layer=deepcopy(pcd_layer)
	last_layer_curve_relative=np.array(layer_curve_relative)
	layer_curve_dh = np.array(layer_curve_dh)

	# curve_i=0
	# total_curve_i = len(layer_curve_dh)
	# ax = plt.figure().add_subplot()
	# for curve_i in range(total_curve_i):
	#     color_dist = plt.get_cmap("rainbow")(float(curve_i)/total_curve_i)
	#     ax.scatter(layer_curve_dh[curve_i,0],layer_curve_dh[curve_i,1],c=color_dist)
	# ax.set_xlabel('Lambda')
	# ax.set_ylabel('dh to Layer N (mm)')
	# ax.set_title("dH Profile")
	# plt.ion()
	# plt.show(block=False)
	
	# curve_i=0
	# total_curve_i = len(layer_curve_height)
	# layer_curve_relative=np.array(layer_curve_relative)
	# lam_curve = calc_lam_cs(layer_curve_relative[:,:3])
	# ax = plt.figure().add_subplot()
	# for curve_i in range(total_curve_i):
	#     color_dist = plt.get_cmap("rainbow")(float(curve_i)/total_curve_i)
	#     ax.scatter(lam_curve[curve_i],layer_curve_height[curve_i],c=color_dist)
	# ax.set_xlabel('Lambda')
	# ax.set_ylabel('Layer N Height (mm)')
	# ax.set_title("Height Profile")
	# plt.show(block=False)

	input("Moving Scan Robot to Home")
	ws.jog_dual(robot2,positioner,[R2_mid,R2_home],[0,q_prev[1]],v=to_home_speed)
	###################### Plot height vs Anticipated Layers ############################
	print("Flame Processed | Plotting Now")


	flame_3d=np.array(flame_3d)
	print(flame_3d.shape)
	#plot the flame 3d
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	try:
		ax.scatter(flame_3d[:,0],flame_3d[:,1],flame_3d[:,2], 'b')
	except IndexError:
		print("No Flame Detected")
	
	
	ax.plot3D(-curve_sliced_relative[:,0], curve_sliced_relative[:,1], curve_sliced_relative[:,2], c='g')
	try:
		for plot_layer in range(layer+2, layer+21, 2):
			curve_sliced_relative=np.loadtxt(data_dir+'curve_sliced_relative/slice'+str(plot_layer)+'_'+str(x)+'.csv',delimiter=',')
			ax.plot3D(-curve_sliced_relative[:,0], curve_sliced_relative[:,1], curve_sliced_relative[:,2], c='r') 
			print("Layer above: ", plot_layer) 
	except FileNotFoundError:
		print("Layers outside of sliced layers")
	try:    
		for plot_layer in range(layer-2, layer-21, -2):
			if plot_layer <=0: raise FileNotFoundError
			curve_sliced_relative=np.loadtxt(data_dir+'curve_sliced_relative/slice'+str(plot_layer)+'_'+str(x)+'.csv',delimiter=',')
			ax.plot3D(-curve_sliced_relative[:,0], curve_sliced_relative[:,1], curve_sliced_relative[:,2], c='b')
			print("Layer below: ", plot_layer)
	except FileNotFoundError: 
		print("No layers prior")    

	#set equal aspect ratio
	try:
		ax.set_box_aspect([np.ptp(flame_3d[:,0]),np.ptp(flame_3d[:,1]),np.ptp(flame_3d[:,2])])
	except IndexError:
		print("No Flame Detected")
		ax.set_aspect('equal')
	ax.set_aspect('equal')
	plt.show()
	layer = int(input(f"Current Layer: {layer} \nEnter Desired Layer Number: "))

	