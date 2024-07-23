import sys, glob, yaml
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

mti_Rpath = np.array([[ -1.,0.,0.],   
                        [ 0.,1.,0.],
                        [0.,0.,-1.]])



## RR sensor objects
rr_sensors = WeldRRSensor(weld_service=None,cam_service=cam_ser,microphone_service=None)

config_dir='../config/'

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


robot=robot_obj('MA2010_A0',def_path='../config/MA2010_A0_robot_default_config.yml',tool_file_path='../config/torch.csv',\
	pulse2deg_file_path='../config/MA2010_A0_pulse2deg_real.csv',d=15)
robot2=robot_obj('MA1440_A0',def_path='../config/MA1440_A0_robot_default_config.yml',tool_file_path='../config/flir.csv',\
	pulse2deg_file_path='../config/MA1440_A0_pulse2deg_real.csv',base_transformation_file='../config/MA1440_pose.csv')
positioner=positioner_obj('D500B',def_path='../config/D500B_robot_extended_config.yml',tool_file_path='../config/positioner_tcp.csv',\
	pulse2deg_file_path='../config/D500B_pulse2deg_real.csv',base_transformation_file='../config/D500B_pose.csv')

client=MotionProgramExecClient()
ws=WeldSend(client)
scan_process = ScanProcess(robot2, positioner)

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

		q_0 = client.getJointAnglesMH(robot.pulse2deg)[0]
		ws.jog_single(robot,[q_0,0,0,0,0,0],4)
		# input("-------Layer Finished-------")

	# send R1_home
	R1_home = np.radians([-60,0,0,0,0,0])
	ws.jog_single(robot, R1_home, v=4)

	################### scan Part ################################
	scan_st = time.time()

	pcd_layer = o3d.geometry.PointCloud()
	layer_curve_relative = []
	layer_curve_dh = []
	for x in range(0, num_sections):
		spg = ScanPathGen(robot2, positioner, scan_stand_off_d, Rz_angle, Ry_angle, bounds_theta)

		curve_sliced_relative=np.loadtxt(data_dir+'curve_sliced_relative/slice'+str(layer)+'_'+str(x)+'.csv',delimiter=',')
		curve_sliced_js=np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(layer)+'_'+str(x)+'.csv',delimiter=',').reshape((-1,6))
		positioner_js=np.loadtxt(data_dir+'curve_sliced_js/D500B_js'+str(layer)+'_'+str(x)+'.csv',delimiter=',')

		rob_js_plan = np.hstack((curve_sliced_js, positioner_js))
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

	