<<<<<<< HEAD
import time, os
from motoman_def import *
from WeldSend import *
from dx200_motion_program_exec_client import *
from RobotRaconteur.Client import *
from weldRRSensor import *

def main():

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
	positioner=positioner_obj('D500B',def_path=config_dir+'D500B_robot_default_config.yml',tool_file_path=config_dir+'positioner_tcp.csv',\
		pulse2deg_file_path=config_dir+'D500B_pulse2deg_real.csv',base_transformation_file=config_dir+'D500B_pose_mocap.csv')

	


	#################################################################robot 1 welding params####################################################################
	R=np.array([[-0.7071, 0.7071, -0.    ],
				[ 0.7071, 0.7071,  0.    ],
				[0.,      0.,     -1.    ]])
	#ystart -850, yend -775
	p_start_base=np.array([1710,-810,-260])
	p_end_base=np.array([1590,-810,-260])
	p_start=np.array([1700,-810,-260])
	p_end=np.array([1600,-810,-260])
	q_seed=np.radians([-35.4291,56.6333,40.5194,4.5177,-52.2505,-11.6546])


	base_feedrate=300
	volume_per_distance=10
	v_layer=15
	feedrate=volume_per_distance*v_layer
	base_layer_height=3
	v_base=5
	layer_height=0.9
	num_layer=30
	#edge params, 1cm left and right
	feedrate_edge=feedrate
	v_edge=v_layer
	q_all=[]
	v_all=[]
	job_offset=450
	cond_all=[]
	primitives=[]


	#################################################################define start pose for 3 robtos####################################################################
	measure_distance=500
	H2010_1440=H_inv(robot2.base_H)
	q_positioner_home=np.array([-15.*np.pi/180.,np.pi/2])
	# p_positioner_home=positioner.fwd(q_positioner_home,world=True).p
	p_positioner_home=np.mean([p_start_base,p_end_base],axis=0)
	p_robot2_proj=p_positioner_home+np.array([0,0,30])
	p2_in_base_frame=np.dot(H2010_1440[:3,:3],p_robot2_proj)+H2010_1440[:3,3]
	v_z=H2010_1440[:3,:3]@np.array([0,-0.96592582628,-0.2588190451]) ###pointing toward positioner's X with 15deg tiltd angle looking down
	v_y=VectorPlaneProjection(np.array([-1,0,0]),v_z)	###FLIR's Y pointing toward 1440's -X in 1440's base frame, projected on v_z's plane
	v_x=np.cross(v_y,v_z)
	p2_in_base_frame=p2_in_base_frame-measure_distance*v_z			###back project measure_distance-mm away from torch
	R2=np.vstack((v_x,v_y,v_z)).T
	q2=robot2.inv(p2_in_base_frame,R2,last_joints=np.zeros(6))[0]

	###jog to start point
	client=MotionProgramExecClient()
	ws=WeldSend(client)
	ws.jog_dual(robot2,positioner,q2,q_positioner_home,v=1)

	####################################Base Layer ####################################
	for i in range(0,2):
		if i%2==0:
			p1=p_start_base+np.array([0,0,i*base_layer_height])
			p2=p_end_base+np.array([0,0,i*base_layer_height])
		else:
			p1=p_end_base+np.array([0,0,i*base_layer_height])
			p2=p_start_base+np.array([0,0,i*base_layer_height])

		
		q_init=robot.inv(p1,R,q_seed)[0]
		q_end=robot.inv(p2,R,q_seed)[0]

		q_all.extend([q_init,q_end])
		v_all.extend([1,v_base])
		primitives.extend(['movej','movel'])
		cond_all.extend([0,int(base_feedrate/10+job_offset)])

	ws.jog_single(robot,robot.inv(p1+np.array([0,0,100]),R,q_seed)[0])
	ws.weld_segment_single(primitives,robot,q_all,v_all,cond_all,arc=True,wait=0.,blocking=True)
	q_all=[]
	v_all=[]
	cond_all=[]
	primitives=[]
	####################################Normal Layer ####################################
	for i in range(2,num_layer+2):
		if i%2==0:
			p1=p_start+np.array([0,0,2*base_layer_height+i*layer_height])
			p2=p_end+np.array([0,0,2*base_layer_height+i*layer_height])
		else:
			p1=p_end+np.array([0,0,2*base_layer_height+i*layer_height])
			p2=p_start+np.array([0,0,2*base_layer_height+i*layer_height])

		p_edge1=p1+(p2-p1)/np.linalg.norm(p2-p1)*10
		p_edge2=p2-(p2-p1)/np.linalg.norm(p2-p1)*10
		q_init=robot.inv(p1,R,q_seed)[0]
		q_end=robot.inv(p2,R,q_seed)[0]
		q_edge1=robot.inv(p_edge1,R,q_seed)[0]
		q_edge2=robot.inv(p_edge2,R,q_seed)[0]
		if i==2:	#if start of first normal layer
			q_all.extend([q_init])
			v_all.extend([1])
			primitives.extend(['movej'])
			cond_all.extend([0])
		
		q_all.extend([q_edge1,q_edge2,q_end])
		v_all.extend([v_edge,v_layer,v_edge])
		primitives.extend(['movel','movel','movel'])
		cond_all.extend([int(feedrate_edge/10+job_offset),int(feedrate/10+job_offset),int(feedrate_edge/10+job_offset)])


	ws.weld_segment_single(primitives,robot,q_all,v_all,cond_all,arc=True,wait=0.,blocking=False)
	##############################################################Log Joint Data####################################################################
	js_recording=[]
	rr_sensors.start_all_sensors()
	start_time=time.time()
	while not(client.state_flag & 0x08 == 0 and time.time()-start_time>1.):
		res, fb_data = client.fb.try_receive_state_sync(client.controller_info, 0.001)
		if res:
			with client._lock:
				client.joint_angle=np.hstack((fb_data.group_state[0].feedback_position,fb_data.group_state[1].feedback_position,fb_data.group_state[2].feedback_position))
				client.state_flag=fb_data.controller_flags
				js_recording.append(np.array([time.time(),fb_data.time]+[fb_data.job_state[0][1]]+client.joint_angle.tolist()))
	rr_sensors.stop_all_sensors()
	client.servoMH(False) #stop the motor


	recorded_dir='../../../recorded_data/ER316L/wallbf_%iipm_v%i_%iipm_v%i/'%(feedrate,v_layer,feedrate_edge,v_edge)
	os.makedirs(recorded_dir,exist_ok=True)
	np.savetxt(recorded_dir+'weld_js_exe.csv',np.array(js_recording),delimiter=',')
	rr_sensors.save_all_sensors(recorded_dir)

if __name__ == '__main__':
=======
import time, os
from motoman_def import *
from WeldSend import *
from dx200_motion_program_exec_client import *
from RobotRaconteur.Client import *
from weldRRSensor import *

def main():

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
	positioner=positioner_obj('D500B',def_path=config_dir+'D500B_robot_default_config.yml',tool_file_path=config_dir+'positioner_tcp.csv',\
		pulse2deg_file_path=config_dir+'D500B_pulse2deg_real.csv',base_transformation_file=config_dir+'D500B_pose_mocap.csv')

	


	#################################################################robot 1 welding params####################################################################
	R=np.array([[-0.7071, 0.7071, -0.    ],
				[ 0.7071, 0.7071,  0.    ],
				[0.,      0.,     -1.    ]])
	#ystart -850, yend -775
	p_start_base=np.array([1710,-810,-260])
	p_end_base=np.array([1590,-810,-260])
	p_start=np.array([1700,-810,-260])
	p_end=np.array([1600,-810,-260])
	q_seed=np.radians([-35.4291,56.6333,40.5194,4.5177,-52.2505,-11.6546])


	base_feedrate=300
	volume_per_distance=10
	v_layer=10
	feedrate=volume_per_distance*v_layer
	base_layer_height=3
	v_base=5
	layer_height=0.3
	num_layer=20
	#edge params, 1cm left and right
	feedrate_edge=feedrate
	v_edge=v_layer
	q_all=[]
	v_all=[]
	job_offset=450
	cond_all=[]
	primitives=[]


	#################################################################define start pose for 3 robtos####################################################################
	measure_distance=500
	H2010_1440=H_inv(robot2.base_H)
	q_positioner_home=np.array([-15.*np.pi/180.,np.pi/2])
	# p_positioner_home=positioner.fwd(q_positioner_home,world=True).p
	p_positioner_home=np.mean([p_start_base,p_end_base],axis=0)
	p_robot2_proj=p_positioner_home+np.array([0,0,30])
	p2_in_base_frame=np.dot(H2010_1440[:3,:3],p_robot2_proj)+H2010_1440[:3,3]
	v_z=H2010_1440[:3,:3]@np.array([0,-0.96592582628,-0.2588190451]) ###pointing toward positioner's X with 15deg tiltd angle looking down
	v_y=VectorPlaneProjection(np.array([-1,0,0]),v_z)	###FLIR's Y pointing toward 1440's -X in 1440's base frame, projected on v_z's plane
	v_x=np.cross(v_y,v_z)
	p2_in_base_frame=p2_in_base_frame-measure_distance*v_z			###back project measure_distance-mm away from torch
	R2=np.vstack((v_x,v_y,v_z)).T
	q2=robot2.inv(p2_in_base_frame,R2,last_joints=np.zeros(6))[0]

	###jog to start point
	client=MotionProgramExecClient()
	ws=WeldSend(client)
	ws.jog_dual(robot2,positioner,q2,q_positioner_home,v=1)

	####################################Base Layer ####################################
	for i in range(0,2):
		if i%2==0:
			p1=p_start_base+np.array([0,0,i*base_layer_height])
			p2=p_end_base+np.array([0,0,i*base_layer_height])
		else:
			p1=p_end_base+np.array([0,0,i*base_layer_height])
			p2=p_start_base+np.array([0,0,i*base_layer_height])

		
		q_init=robot.inv(p1,R,q_seed)[0]
		q_end=robot.inv(p2,R,q_seed)[0]

		q_all.extend([q_init,q_end])
		v_all.extend([1,v_base])
		primitives.extend(['movej','movel'])
		cond_all.extend([0,int(base_feedrate/10+job_offset)])

	ws.jog_single(robot,robot.inv(p1+np.array([0,0,100]),R,q_seed)[0])
	ws.weld_segment_single(primitives,robot,q_all,v_all,cond_all,arc=True,wait=0.,blocking=True)
	q_all=[]
	v_all=[]
	cond_all=[]
	primitives=[]
	####################################Normal Layer ####################################
	for i in range(2,num_layer+2):
		if i%2==0:
			p1=p_start+np.array([0,0,2*base_layer_height+i*layer_height])
			p2=p_end+np.array([0,0,2*base_layer_height+i*layer_height])
		else:
			p1=p_end+np.array([0,0,2*base_layer_height+i*layer_height])
			p2=p_start+np.array([0,0,2*base_layer_height+i*layer_height])

		p_edge1=p1+(p2-p1)/np.linalg.norm(p2-p1)*10
		p_edge2=p2-(p2-p1)/np.linalg.norm(p2-p1)*10
		q_init=robot.inv(p1,R,q_seed)[0]
		q_end=robot.inv(p2,R,q_seed)[0]
		q_edge1=robot.inv(p_edge1,R,q_seed)[0]
		q_edge2=robot.inv(p_edge2,R,q_seed)[0]
		if i==2:	#if start of first normal layer
			q_all.extend([q_init])
			v_all.extend([1])
			primitives.extend(['movej'])
			cond_all.extend([0])
		
		q_all.extend([q_edge1,q_edge2,q_end])
		v_all.extend([v_edge,v_layer,v_edge])
		primitives.extend(['movel','movel','movel'])
		cond_all.extend([int(feedrate_edge/10+job_offset),int(feedrate/10+job_offset),int(feedrate_edge/10+job_offset)])

	##############################################################Log & Execution####################################################################
	rr_sensors.start_all_sensors()
	global_ts,robot_ts,joint_recording,job_line,_ = ws.weld_segment_single(primitives,robot,q_all,v_all,cond_all,arc=True,wait=0.)
	rr_sensors.stop_all_sensors()


	# recorded_dir='../../../recorded_data/ER316L/wallbf_%iipm_v%i_%iipm_v%i/'%(feedrate,v_layer,feedrate_edge,v_edge)
	# os.makedirs(recorded_dir,exist_ok=True)
	# np.savetxt(recorded_dir+'weld_js_exe.csv',np.hstack((global_ts.reshape(-1, 1),robot_ts.reshape(-1, 1),job_line.reshape(-1, 1),joint_recording)),delimiter=',')
	# rr_sensors.save_all_sensors(recorded_dir)

if __name__ == '__main__':
>>>>>>> upstream/main
	main()