import sys, glob
sys.path.append('../toolbox/')
from robot_def import *
from lambda_calc import *
from multi_robot import *
from dx200_motion_program_exec_client import *
from WeldSend import *
from RobotRaconteur.Client import *
import weld_dh2v
import matplotlib.pyplot as plt


dataset='bent_thick_wall/'
sliced_alg='slice_ER_4043/'
data_dir='../data/'+dataset+sliced_alg
with open(data_dir+'slicing.yml', 'r') as file:
	slicing_meta = yaml.safe_load(file)
# recorded_dir='recorded_data/cup_ER316L/'
waypoint_distance=1.5
layer_width_num=int(3/slicing_meta['line_resolution'])


robot=robot_obj('MA2010_A0',def_path='../config/MA2010_A0_robot_default_config.yml',tool_file_path='../config/torch.csv',\
	pulse2deg_file_path='../config/MA2010_A0_pulse2deg_real.csv',d=15)
positioner=positioner_obj('D500B',def_path='../config/D500B_robot_extended_config.yml',tool_file_path='../config/positioner_tcp.csv',\
	pulse2deg_file_path='../config/D500B_pulse2deg_real.csv',base_transformation_file='../config/D500B_pose.csv')

client=MotionProgramExecClient()
ws=WeldSend(client)

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


##########################################BASE layer welding############################################
num_layer_start=int(0*nominal_slice_increment)	###modify layer num here
num_layer_end=int(1*nominal_slice_increment)
#q_prev=client.getJointAnglesDB(positioner.pulse2deg)
q_prev=np.array([9.53E-02,-2.71E+00])	###for motosim tests only

for layer in range(num_layer_start,num_layer_end,nominal_slice_increment):
	mp=MotionProgram(ROBOT_CHOICE='RB1',ROBOT_CHOICE2='ST1',pulse2deg=robot.pulse2deg,pulse2deg_2=positioner.pulse2deg, tool_num = 12)

	num_sections=len(glob.glob(data_dir+'curve_sliced_relative/slice'+str(layer)+'_*.csv'))

	####################DETERMINE CURVE ORDER##############################################
	for x in range(0,num_sections,layer_width_num):
		curve_sliced_js=np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(layer)+'_'+str(x)+'.csv',delimiter=',').reshape((-1,6))
		if len(curve_sliced_js)<2:
			continue
		positioner_js=np.loadtxt(data_dir+'curve_sliced_js/D500B_js'+str(layer)+'_'+str(x)+'.csv',delimiter=',')
		curve_sliced_relative=np.loadtxt(data_dir+'curve_sliced_relative/slice'+str(layer)+'_'+str(x)+'.csv',delimiter=',')

		lam1=calc_lam_js(curve_sliced_js,robot)
		lam2=calc_lam_js(positioner_js,positioner)
		lam_relative=calc_lam_cs(curve_sliced_relative)

		num_points_layer=max(2,int(lam_relative[-1]/waypoint_distance))

		###find which end to start depending on how close to joint limit
		if positioner.upper_limit[1]-q_prev[1]>q_prev[1]-positioner.lower_limit[1]:
			breakpoints=np.linspace(0,len(curve_sliced_js)-1,num=num_points_layer).astype(int)
		else:
			breakpoints=np.linspace(len(curve_sliced_js)-1,0,num=num_points_layer).astype(int)

		

		#s1_all,s2_all=calc_individual_speed(base_vd,lam1,lam2,lam_relative,breakpoints)

		q1_all=[curve_sliced_js[breakpoints[0]]]
		q2_all=[positioner_js[breakpoints[0]]]
		v1_all=[2]
		v2_all=[10]
		primitives=['movej']
		for j in range(1,len(breakpoints)):
			q1_all.append(curve_sliced_js[breakpoints[j]])
			q2_all.append(positioner_js[breakpoints[j]])
			v1_all.append(max(base_vd,0.1))
			positioner_w=vd_relative/np.linalg.norm(curve_sliced_relative[breakpoints[j]][:2])
			v2_all.append(min(100,100*positioner_w/positioner.joint_vel_limit[1]))
			primitives.append('movel')

		q_prev=positioner_js[breakpoints[-1]]
		timestamp_robot,joint_recording,job_line,_=ws.weld_segment_dual(primitives,robot,positioner,q1_all,q2_all,v1_all,v2_all,cond_all=[int(base_feedrate_cmd/10)+job_offset],arc=True)
		q_0 = client.getJointAnglesMH(robot.pulse2deg)[0]
		ws.jog_single(robot,[q_0,0,0,0,0,0],4)
		print("layer: ", layer)
		input("-------Base Layer Finished-------")




###########################################layer welding############################################
num_layer_start=int(1*nominal_slice_increment)	###modify layer num here
num_layer_end=min(70*nominal_slice_increment,slicing_meta['num_layers'])

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

start_dir = True # Alternate direction that the layer starts from

for layer in range(num_layer_start,num_layer_end,nominal_slice_increment):
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

		lam1=calc_lam_js(curve_sliced_js,robot)
		lam2=calc_lam_js(positioner_js,positioner)
		lam_relative=calc_lam_cs(curve_sliced_relative)
		num_points_layer=max(2,int(lam_relative[-1]/waypoint_distance))
		breakpoints=np.linspace(0,len(curve_sliced_js)-1,num=num_points_layer).astype(int)
		'''###find which end to start depending on how close to joint limit
		if positioner.upper_limit[1]-q_prev[1]>q_prev[1]-positioner.lower_limit[1]:
			breakpoints=np.linspace(0,len(curve_sliced_js)-1,num=num_points_layer).astype(int)
		else:
			breakpoints=np.linspace(len(curve_sliced_js)-1,0,num=num_points_layer).astype(int)'''
		
		###alternate the start point on different ends

		# if start_dir: breakpoints=np.linspace(0,len(curve_sliced_js)-1,num=num_points_layer).astype(int)
		# else:
		# 	breakpoints=np.linspace(len(curve_sliced_js)-1,0,num=num_points_layer).astype(int)
		# start_dir = not start_dir
		
		###########################################velocity profile#########################################
		dh_max = slicing_meta['dh_max']
		dh_min = slicing_meta['dh_min']
		with open(data_dir+'vel_profile.pkl', 'rb') as file:
			velocity_profile = pickle.load(file)
		
		# height_profile = np.zeros(len(curve_sliced_js))
		# height_profile = np.linspace(dh_max, dh_min, len(curve_sliced_js))
		# velocity_profile = weld_dh2v.dh2v_loglog(height_profile, feedrate_cmd, 'ER_4043')

		print("velocity profile: ",velocity_profile)
		#pickle.dump(velocity_profile, open(data_dir+'layer_velocity_profile.pickle','wb'))
		###move to intermidieate waypoint for collision avoidance if multiple section
		if num_sections!=num_sections_prev:
			waypoint_pose=robot.fwd(curve_sliced_js[breakpoints[0]])
			waypoint_pose.p[-1]+=50
			q1=robot.inv(waypoint_pose.p,waypoint_pose.R,curve_sliced_js[breakpoints[0]])[0]
			q2=positioner_js[breakpoints[0]]
			ws.jog_dual(robot,positioner,q1,q2,v=100)
		#print("breakpoints: ",breakpoints)
		q1_all=[curve_sliced_js[breakpoints[0]]]
		q2_all=[positioner_js[breakpoints[0]]]
		v1_all=[2]
		v2_all=[10]
		primitives=['movej']
		for j in range(0,len(breakpoints)):
			q1_all.append(curve_sliced_js[breakpoints[j]])
			q2_all.append(positioner_js[breakpoints[j]])
			v1_all.append(max(velocity_profile[breakpoints[j]],0.1))# remove speed modifier before acutal run
			
			positioner_w=vd_relative/np.linalg.norm(curve_sliced_relative[breakpoints[j]][:2])
			v2_all.append(min(100,100*positioner_w/positioner.joint_vel_limit[1]))
			primitives.append('movel')
		q_prev=positioner_js[breakpoints[-1]]
		print('V1: ', v1_all)
		print("layer: ", layer)
		timestamp_robot,joint_recording,job_line,_=ws.weld_segment_dual(primitives,robot,positioner,q1_all,q2_all,v1_all,v2_all,cond_all=[int(feedrate_cmd/10)+job_offset],arc=True)
		q_0 = client.getJointAnglesMH(robot.pulse2deg)[0]
		ws.jog_single(robot,[q_0,0,0,0,0,0],4)
		input(f"-------Layer {layer} Finished-------")
