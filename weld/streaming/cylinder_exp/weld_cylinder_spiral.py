import sys, time, os
from motoman_def import *
from lambda_calc import *
from RobotRaconteur.Client import *
from weldRRSensor import *
from dual_robot import *
from traj_manipulation import *
from StreamingSend import *


def main():

	dataset='cylinder/'
	sliced_alg='dense_slice/'
	data_dir='../../data/'+dataset+sliced_alg
	with open(data_dir+'slicing.yml', 'r') as file:
		slicing_meta = yaml.safe_load(file)


	##############################################################SENSORS####################################################################
	# weld state logging
	# weld_ser = RRN.SubscribeService('rr+tcp://192.168.55.10:60823?service=welder')
	cam_ser=RRN.ConnectService('rr+tcp://localhost:60827/?service=camera')
	# mic_ser = RRN.ConnectService('rr+tcp://192.168.55.20:60828?service=microphone')
	## RR sensor objects
	rr_sensors = WeldRRSensor(weld_service=None,cam_service=cam_ser,microphone_service=None)

	##############################################################Robot####################################################################
	###robot kinematics def
	config_dir='../../config/'
	robot=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'torch.csv',\
		pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',d=15)
	robot2=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'flir.csv',\
		pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv',base_transformation_file=config_dir+'MA1440_pose.csv')
	positioner=positioner_obj('D500B',def_path=config_dir+'D500B_robot_extended_config.yml',tool_file_path=config_dir+'positioner_tcp.csv',\
		pulse2deg_file_path=config_dir+'D500B_pulse2deg_real.csv',base_transformation_file=config_dir+'D500B_pose_mocap.csv')

	###define start pose for 3 robtos
	measure_distance=500
	H2010_1440=H_inv(robot2.base_H)
	q_positioner_home=np.array([-15.*np.pi/180.,np.pi/2])
	# p_positioner_home=positioner.fwd(q_positioner_home,world=True).p
	rob1_js=np.loadtxt(data_dir+'curve_sliced_js/MA2010_js0_0.csv',delimiter=',')
	positioner_js=np.loadtxt(data_dir+'curve_sliced_js/D500B_js0_0.csv',delimiter=',')
	p_positioner_home=np.mean([robot.fwd(rob1_js[0]).p,robot.fwd(rob1_js[-1]).p],axis=0)
	p_robot2_proj=p_positioner_home+np.array([0,0,50])
	p2_in_base_frame=np.dot(H2010_1440[:3,:3],p_robot2_proj)+H2010_1440[:3,3]
	v_z=H2010_1440[:3,:3]@np.array([0,-0.96592582628,-0.2588190451]) ###pointing toward positioner's X with 15deg tiltd angle looking down
	v_y=VectorPlaneProjection(np.array([-1,0,0]),v_z)	###FLIR's Y pointing toward 1440's -X in 1440's base frame, projected on v_z's plane
	v_x=np.cross(v_y,v_z)
	p2_in_base_frame=p2_in_base_frame-measure_distance*v_z			###back project measure_distance-mm away from torch
	R2=np.vstack((v_x,v_y,v_z)).T
	q2=robot2.inv(p2_in_base_frame,R2,last_joints=np.zeros(6))[0]

	########################################################RR FRONIUS########################################################
	weld_arcon=True
	if weld_arcon:
		fronius_sub=RRN.SubscribeService('rr+tcp://192.168.55.21:60823?service=welder')
		fronius_client = fronius_sub.GetDefaultClientWait(1)      #connect, timeout=30s
		hflags_const = RRN.GetConstants("experimental.fronius", fronius_client)["WelderStateHighFlags"]
		fronius_client.prepare_welder()
	
	########################################################RR STREAMING########################################################
	RR_robot_sub = RRN.SubscribeService('rr+tcp://192.168.55.12:59945?service=robot')
	point_distance=0.04		###STREAMING POINT INTERPOLATED DISTANCE
	SS=StreamingSend(RR_robot_sub,streaming_rate=125.)

	#################################################################robot 1 welding params####################################################################
	R=np.array([[-0.7071, 0.7071, -0.    ],
				[ 0.7071, 0.7071,  0.    ],
				[0.,      0.,     -1.    ]])

	
	base_feedrate=300
	volume_per_distance=10
	v_layer=10
	feedrate=volume_per_distance*v_layer
	base_layer_height=5.5
	v_base=5
	layer_height=1.1
	num_layer=30
	q_all=[]
	v_all=[]
	job_offset=450


	nominal_slice_increment=int(layer_height/slicing_meta['line_resolution'])
	base_slice_increment=int(base_layer_height/slicing_meta['line_resolution'])


	q_positioner_prev=SS.q_cur[-2:]
	# num2p=np.round((q_positioner_prev-positioner_js[0])/(2*np.pi))
	# positioner_js+=num2p*2*np.pi
	
	
	
	# #####################################################BASE LAYER##########################################################################################
	# ###PRELOAD ALL SLICES TO SAVE INPROCESS TIME
	# rob1_js_all_slices=[]
	# positioner_js_all_slices=[]
	# for i in range(0,2*base_slice_increment):
	# 	# rob1_js_all_slices.append(np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(i)+'_0.csv',delimiter=','))
	# 	# positioner_js_all_slices.append(np.loadtxt(data_dir+'curve_sliced_js/D500B_js'+str(i)+'_0.csv',delimiter=','))

	# 	###spiral rotation direction
	# 	rob1_js_all_slices.append(np.flip(np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(i)+'_0.csv',delimiter=','),axis=0))
	# 	positioner_js_all_slices.append(np.flip(np.loadtxt(data_dir+'curve_sliced_js/D500B_js'+str(i)+'_0.csv',delimiter=','),axis=0))

	# print("PRELOAD FINISHED")

	# num_layer_end=2*base_slice_increment
	# for slice_num in range(0,num_layer_end,base_slice_increment):

	# 	####################DETERMINE CURVE ORDER##############################################
	# 	x=0
	# 	rob1_js=copy.deepcopy(rob1_js_all_slices[slice_num])
	# 	positioner_js=copy.deepcopy(positioner_js_all_slices[slice_num])
	# 	curve_sliced_relative=np.loadtxt(data_dir+'curve_sliced_relative/slice'+str(slice_num)+'_'+str(x)+'.csv',delimiter=',')
	# 	if positioner_js.shape==(2,) and rob1_js.shape==(6,):
	# 		continue
		
	# 	###TRJAECTORY WARPING
	# 	if slice_num>0:
	# 		rob1_js_prev=copy.deepcopy(rob1_js_all_slices[slice_num-base_slice_increment])
	# 		positioner_js_prev=copy.deepcopy(positioner_js_all_slices[slice_num-base_slice_increment])
	# 		rob1_js,positioner_js=warp_traj2(rob1_js,positioner_js,rob1_js_prev,positioner_js_prev,reversed=True)
	# 	if slice_num<num_layer_end-base_slice_increment:
	# 		rob1_js_next=copy.deepcopy(rob1_js_all_slices[slice_num+base_slice_increment])
	# 		positioner_js_next=copy.deepcopy(positioner_js_all_slices[slice_num+base_slice_increment])
	# 		rob1_js,positioner_js=warp_traj2(rob1_js,positioner_js,rob1_js_next,positioner_js_next,reversed=False)
				
		
			
	# 	lam_relative=calc_lam_cs(curve_sliced_relative)
	# 	lam_relative_dense=np.linspace(0,lam_relative[-1],num=int(lam_relative[-1]/point_distance))
	# 	rob1_js_dense=interp1d(lam_relative,rob1_js,kind='cubic',axis=0)(lam_relative_dense)
	# 	positioner_js_dense=interp1d(lam_relative,positioner_js,kind='cubic',axis=0)(lam_relative_dense)
	# 	breakpoints=SS.get_breakpoints(lam_relative_dense,v_base)

	# 	###find closest %2pi
	# 	num2p=np.round((q_positioner_prev-positioner_js_dense[0])/(2*np.pi))
	# 	positioner_js_dense+=num2p*2*np.pi
		
	# 	###formulate streaming joint angles
	# 	q_all.extend(np.hstack((rob1_js_dense[breakpoints],[q2]*len(breakpoints),positioner_js_dense[breakpoints])))
		
	# 	q_positioner_prev=copy.deepcopy(positioner_js_dense[-1])

	# q_all=np.array(q_all)

	# ###jog to start point
	# print("BASELAYER CALCULATION FINISHED")
	# SS.jog2q(q_all[0])
	# ##############################################################Base Layers Welding####################################################################
	# if weld_arcon:
	# 	fronius_client.job_number = int(base_feedrate/10+job_offset)
	# 	fronius_client.start_weld()
	# for i in range(len(q_all)):
	# 	SS.position_cmd(q_all[i],time.perf_counter())
	# if weld_arcon:
	# 	fronius_client.stop_weld()
	# print("BASELAYER WELDING FINISHED")




	###############################################################################################################################################################
	###############################################################################################################################################################
	#####################################################LAYER Welding##########################################################################################
	###PRELOAD ALL SLICES TO SAVE INPROCESS TIME
	rob1_js_all_slices=[]
	positioner_js_all_slices=[]
	for i in range(slicing_meta['num_layers']):
		# rob1_js_all_slices.append(np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(i)+'_0.csv',delimiter=','))
		# positioner_js_all_slices.append(np.loadtxt(data_dir+'curve_sliced_js/D500B_js'+str(i)+'_0.csv',delimiter=','))
		###spiral rotation direction
		rob1_js_all_slices.append(np.flip(np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(i)+'_0.csv',delimiter=','),axis=0))
		positioner_js_all_slices.append(np.flip(np.loadtxt(data_dir+'curve_sliced_js/D500B_js'+str(i)+'_0.csv',delimiter=','),axis=0))

	print("PRELOAD FINISHED")

	num_layer_start=int(2*base_slice_increment)
	num_layer_end=slicing_meta['num_layers']-1
	layer_counts=0
	for slice_num in range(num_layer_start,num_layer_end,nominal_slice_increment):

		####################DETERMINE CURVE ORDER##############################################
		x=0
		rob1_js=copy.deepcopy(rob1_js_all_slices[slice_num])
		positioner_js=copy.deepcopy(positioner_js_all_slices[slice_num])
		curve_sliced_relative=np.loadtxt(data_dir+'curve_sliced_relative/slice'+str(slice_num)+'_'+str(x)+'.csv',delimiter=',')
		if positioner_js.shape==(2,) and rob1_js.shape==(6,):
			continue
		
		###TRJAECTORY WARPING
		if slice_num>num_layer_start:
			rob1_js_prev=copy.deepcopy(rob1_js_all_slices[slice_num-nominal_slice_increment])
			positioner_js_prev=copy.deepcopy(positioner_js_all_slices[slice_num-nominal_slice_increment])
			rob1_js,positioner_js=warp_traj2(rob1_js,positioner_js,rob1_js_prev,positioner_js_prev,reversed=True)
		if slice_num<num_layer_end-nominal_slice_increment:
			rob1_js_next=copy.deepcopy(rob1_js_all_slices[slice_num+nominal_slice_increment])
			positioner_js_next=copy.deepcopy(positioner_js_all_slices[slice_num+nominal_slice_increment])
			rob1_js,positioner_js=warp_traj2(rob1_js,positioner_js,rob1_js_next,positioner_js_next,reversed=False)
				
		
			
		lam_relative=calc_lam_cs(curve_sliced_relative)
		lam_relative_dense=np.linspace(0,lam_relative[-1],num=int(lam_relative[-1]/point_distance))
		rob1_js_dense=interp1d(lam_relative,rob1_js,kind='cubic',axis=0)(lam_relative_dense)
		positioner_js_dense=interp1d(lam_relative,positioner_js,kind='cubic',axis=0)(lam_relative_dense)
		breakpoints=SS.get_breakpoints(lam_relative_dense,v_layer)

		###find closest %2pi
		num2p=np.round((q_positioner_prev-positioner_js_dense[0])/(2*np.pi))
		positioner_js_dense+=num2p*2*np.pi
		
		###formulate streaming joint angles
		q_all.extend(np.hstack((rob1_js_dense[breakpoints],[q2]*len(breakpoints),positioner_js_dense[breakpoints])))
				
		q_positioner_prev=copy.deepcopy(positioner_js_dense[-1])
		layer_counts+=1
		if layer_counts>=num_layer:
			break

	q_all=np.array(q_all)
	print("Layer CALCULATION FINISHED")
	
	###jog to start point
	SS.jog2q(q_all[0])

	############################################################Welding Normal Layers ####################################################################

	rr_sensors.start_all_sensors()
	SS.start_recording()
	if weld_arcon:
		fronius_client.job_number = int(feedrate/10+job_offset)
		fronius_client.start_weld()

	for i in range(len(q_all)):
		SS.position_cmd(q_all[i],time.perf_counter())
	
	if weld_arcon:
		fronius_client.stop_weld()
	rr_sensors.stop_all_sensors()
	js_recording = SS.stop_recording()


	recorded_dir='../../../recorded_data/streaming/ER316L/cylinderspiral_%iipm_v%i/'%(feedrate,v_layer)
	os.makedirs(recorded_dir,exist_ok=True)
	np.savetxt(recorded_dir+'weld_js_exe.csv',np.array(js_recording),delimiter=',')
	np.savetxt(recorded_dir+'weld_js_cmd.csv',np.array(q_all),delimiter=',')
	rr_sensors.save_all_sensors(recorded_dir)


if __name__ == '__main__':
	main()