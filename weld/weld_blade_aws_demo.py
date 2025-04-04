import sys, glob
from motoman_def import *
from lambda_calc import *
from dual_robot import *
from dx200_motion_program_exec_client import *
from WeldSend import *
from scipy.ndimage import gaussian_filter1d


def extend_simple(curve_sliced_js,positioner_js,curve_sliced_relative,lam_relative,d=10):
	pose0=robot.fwd(curve_sliced_js[0])
	pose1=robot.fwd(curve_sliced_js[1])
	vd_start=pose0.p-pose1.p
	vd_start/=np.linalg.norm(vd_start)
	q_start=robot.inv(pose0.p+d*vd_start,pose0.R,curve_sliced_js[0])[0]

	pose_1=robot.fwd(curve_sliced_js[-1])
	pose_2=robot.fwd(curve_sliced_js[-2])
	vd_end=pose_1.p-pose_2.p
	vd_end/=np.linalg.norm(vd_end)
	q_end=robot.inv(pose_1.p+d*vd_end,pose0.R,curve_sliced_js[-1])[0]

	return q_start, q_end


dataset='blade0.1/'
sliced_alg='dense_slice/'
data_dir='../../geometry_data/'+dataset+sliced_alg
with open(data_dir+'slicing.yml', 'r') as file:
	slicing_meta = yaml.safe_load(file)

waypoint_distance=5 	###waypoint separation
layer_width_num=int(4/slicing_meta['line_resolution'])

weld_arcon=False
# #######################################ER4043########################################################
job_offset=200
vd_relative=8
feedrate_cmd=110
base_vd_relative=3
base_feedrate_cmd=300
layer_height_num=int(1.15/slicing_meta['line_resolution'])

#######################################ER70S-6########################################################
# job_offset=300
# vd_relative=7
# feedrate_cmd=120
# base_vd_relative=5
# base_feedrate_cmd=300
# layer_height_num=int(1.04/slicing_meta['line_resolution'])

# #######################################ER316L THIN########################################################
# job_offset=400
# vd_relative=7
# feedrate_cmd=250
# base_vd_relative=5
# base_feedrate_cmd=250
# layer_height_num=int(1.5/slicing_meta['line_resolution'])

#######################################ER316L THICK########################################################
# job_offset=450
# vd_relative=7
# feedrate_cmd=140
# base_vd_relative=5
# base_feedrate_cmd=250
# layer_height_num=int(1.4/slicing_meta['line_resolution'])


robot=robot_obj('MA2010_A0',def_path='../config/MA2010_A0_robot_default_config.yml',tool_file_path='../config/torch.csv',\
	pulse2deg_file_path='../config/MA2010_A0_pulse2deg_real.csv',d=15)
#TODO: change to fujicam tool file
robot_fuji=robot_obj('MA2010_A0',def_path='../config/MA2010_A0_robot_default_config.yml',tool_file_path='../config/fujicam.csv',\
	pulse2deg_file_path='../config/MA2010_A0_pulse2deg_real.csv')
robot2=robot_obj('MA1440_A0',def_path='../config/MA1440_A0_robot_default_config.yml',tool_file_path='../config/flir.csv',\
	pulse2deg_file_path='../config/MA1440_A0_pulse2deg_real.csv',base_transformation_file='../config/MA1440_pose.csv')
positioner=positioner_obj('D500B',def_path='../config/D500B_robot_default_config.yml',tool_file_path='../config/positioner_tcp.csv',\
	pulse2deg_file_path='../config/D500B_pulse2deg_real.csv',base_transformation_file='../config/D500B_pose.csv')

mp=MotionProgram(ROBOT_CHOICE='RB1',ROBOT_CHOICE2='ST1',pulse2deg=robot.pulse2deg,pulse2deg_2=positioner.pulse2deg, tool_num = 12)
client=MotionProgramExecClient()
ws=WeldSend(client)

###########################################base layer welding############################################
q1_all=[]
positioner_all=[]
q2_all=[]
v1_all=[]
cond_all=[]
primitives=[]

num_baselayer=2
###far end
# q_prev=np.array([-3.250117036572426343e-01,8.578573591937989073e-01,5.007170842167016911e-01,4.055312631131529622e-01,-1.119412117298938414e+00,-1.407346519936740536e+00])
###close end
q_prev=np.array([-3.791544713877046391e-01,7.156749523014762637e-01,2.756772964158371586e-01,2.106493295914119712e-01,-7.865937103692784982e-01,-5.293956242391706368e-01])

mp=MotionProgram(ROBOT_CHOICE='RB1',ROBOT_CHOICE2='ST1',pulse2deg=robot.pulse2deg,pulse2deg_2=positioner.pulse2deg, tool_num = 12)
for base_layer in range(num_baselayer):
	num_sections=len(glob.glob(data_dir+'curve_sliced_relative/base_slice'+str(base_layer)+'_*.csv'))
	for x in range(num_sections):
		rob1_js=np.loadtxt(data_dir+'curve_sliced_js/MA2010_base_js'+str(base_layer)+'_'+str(x)+'.csv',delimiter=',')
		positioner_js=np.loadtxt(data_dir+'curve_sliced_js/D500B_base_js'+str(base_layer)+'_'+str(x)+'.csv',delimiter=',')
		curve_sliced_relative=np.loadtxt(data_dir+'curve_sliced_relative/base_slice'+str(base_layer)+'_'+str(x)+'.csv',delimiter=',')

		lam1=calc_lam_js(rob1_js,robot)
		lam2=calc_lam_js(positioner_js,positioner)
		lam_relative=calc_lam_cs(curve_sliced_relative)

		q_start,q_end=extend_simple(rob1_js,positioner_js,curve_sliced_relative,lam_relative,d=10)

		num_points_layer=max(2,int(lam_relative[-1]/waypoint_distance))
		###find which end to start
		if np.linalg.norm(q_prev-rob1_js[0])<np.linalg.norm(q_prev-rob1_js[-1]):
			breakpoints=np.linspace(0,len(rob1_js)-1,num=num_points_layer).astype(int)
		else:
			temp=copy.deepcopy(q_start)
			q_start=copy.deepcopy(q_end)
			q_end=temp
			breakpoints=np.linspace(len(rob1_js)-1,0,num=num_points_layer).astype(int)

		s1_all,_=calc_individual_speed(base_vd_relative,lam1,lam2,lam_relative,breakpoints)

		primitives.extend(['movej']+['movel']*(num_points_layer+1))
		q1_all.extend([q_start]+rob1_js[breakpoints].tolist()+[q_end])
		q2_all.extend([positioner_js[breakpoints[0]]]+positioner_js[breakpoints].tolist()+[positioner_js[breakpoints[-1]]])
		v1_all.extend([1]+[s1_all[0]]+s1_all+[s1_all[-1]])
		cond_all.extend([0]+[int(base_feedrate_cmd/10+job_offset)]*(num_points_layer+1))					###extended baselayer welding
		

		q_prev=rob1_js[breakpoints[-1]]


ws.weld_segment_dual(primitives,robot,positioner,q1_all,q2_all,v1_all,10*np.ones(len(v1_all)),cond_all,arc=weld_arcon)

###########################################layer welding############################################
###far end
# q_prev=np.array([-3.250117036572426343e-01,8.578573591937989073e-01,5.007170842167016911e-01,4.055312631131529622e-01,-1.119412117298938414e+00,-1.407346519936740536e+00])
###close end
# q_prev=np.array([-3.791544713877046391e-01,7.156749523014762637e-01,2.756772964158371586e-01,2.106493295914119712e-01,-7.865937103692784982e-01,-5.293956242391706368e-01])
q_prev=np.array([-0.35223283,  0.69407904,  0.30906425,  0.40782731, -1.21226259, -1.72635375])
# q_prev=client.getJointAnglesMH(robot.pulse2deg)


q1_all=[]
positioner_all=[]
q2_all=[]
v1_all=[]
cond_all=[]
primitives=[]


num_layer_start=int(0*layer_height_num)
num_layer_end=int(70*layer_height_num)
curve_slices_pc=[]
num_sections=1
for layer in range(num_layer_start,num_layer_end,layer_height_num):
	num_sections_prev=num_sections
	num_sections=len(glob.glob(data_dir+'curve_sliced_relative/slice'+str(layer)+'_*.csv'))


	###############DETERMINE SECTION ORDER###########################
	if num_sections==0:
		break
	if num_sections==1:
		sections=[0]
	else:
		endpoints=[]
		rob1_js_first=np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(layer)+'_0.csv',delimiter=',')
		rob1_js_last=np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(layer)+'_'+str(num_sections-1)+'.csv',delimiter=',')
		endpoints=np.array([rob1_js_first[0],rob1_js_first[-1],rob1_js_last[0],rob1_js_last[-1]])
		clost_idx=np.argmin(np.linalg.norm(endpoints[:,:5]-q_prev[:5],axis=1))
		if clost_idx>1:
			sections=reversed(range(num_sections))
		else:
			sections=range(num_sections)

	####################DETERMINE CURVE ORDER##############################################
	for x in sections:
		rob1_js=np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(layer)+'_'+str(x)+'.csv',delimiter=',')
		rob2_js=np.loadtxt(data_dir+'curve_sliced_js/MA1440_js'+str(layer)+'_'+str(x)+'.csv',delimiter=',')
		positioner_js=np.loadtxt(data_dir+'curve_sliced_js/D500B_js'+str(layer)+'_'+str(x)+'.csv',delimiter=',')
		curve_sliced_relative=np.loadtxt(data_dir+'curve_sliced_relative/slice'+str(layer)+'_'+str(x)+'.csv',delimiter=',')
		curve_slices_pc.append(curve_sliced_relative[:,:3])

		
		lam1=calc_lam_js(rob1_js,robot)
		lam2=calc_lam_js(positioner_js,positioner)
		lam_relative=calc_lam_cs(curve_sliced_relative)

		num_points_layer=max(2,int(lam_relative[-1]/waypoint_distance))

		###find which end to start
		if np.linalg.norm(q_prev[:3]-rob1_js[0][:3])<np.linalg.norm(q_prev[:3]-rob1_js[-1][:3]):
			breakpoints=np.linspace(0,len(rob1_js)-1,num=num_points_layer).astype(int)
		else:
			breakpoints=np.linspace(len(rob1_js)-1,0,num=num_points_layer).astype(int)

		# print(breakpoints)
		s1_all,_=calc_individual_speed(vd_relative,lam1,lam2,lam_relative,breakpoints)
		
		###move to intermidieate waypoint for collision avoidance if multiple section
		if num_sections>1 or num_sections<num_sections_prev or layer==int(1*layer_height_num):
			waypoint_pose=robot.fwd(rob1_js[breakpoints[0]])
			waypoint_pose.p[-1]+=30
			waypoint_q=robot.inv(waypoint_pose.p,waypoint_pose.R,rob1_js[breakpoints[0]])[0]

			q1_all.append(waypoint_q)
			q2_all.append(rob2_js[breakpoints[0]])
			positioner_all.append(positioner_js[breakpoints[0]])
			v1_all.append(8)
			cond_all.append(0)
			primitives.append('movej')

		q1_all.extend(rob1_js[breakpoints].tolist())
		q2_all.extend(rob2_js[breakpoints].tolist())
		positioner_all.extend(positioner_js[breakpoints].tolist())
		v1_all.extend([8]+s1_all)
		cond_all.extend([0]+[int(feedrate_cmd/10+job_offset)]*(num_points_layer-1))
		primitives.extend(['movej']+['movel']*(num_points_layer-1))


		q_prev=copy.deepcopy(q1_all[-1])
	

ws.weld_segment_tri(primitives,robot,positioner,robot2,q1_all,positioner_all,q2_all,v1_all,v1_all,cond_all,arc=weld_arcon,blocking=False)
####################################################Real Time PCD Update############################################
import open3d as o3d
from RobotRaconteur.Client import *     #import RR client library
# time.sleep(5)
curve_sliced_pc=np.vstack(curve_slices_pc)

#RR client setup, connect to turtle service
url='rr+tcp://localhost:12181/?service=fujicam'
#take url from command line
if (len(sys.argv)>=2):
	url=sys.argv[1]

########subscription mode
def connect_failed(s, client_id, url, err):
	print ("Client connect failed: " + str(client_id.NodeID) + " url: " + str(url) + " error: " + str(err))

sub=RRN.SubscribeService(url)
obj = sub.GetDefaultClientWait(2)		#connect, timeout=2s
scan_change=sub.SubscribeWire("lineProfile")
sub.ClientConnectFailed += connect_failed

# get matplotlib color map
cmap = plt.get_cmap('jet')
estimated_max_height = 65

all_points=np.loadtxt(data_dir+'curve_sliced_relative/slice0_0.csv',delimiter=',')[:,:3]
all_points_color = np.tile(cmap(np.mean(all_points[:,2]) / estimated_max_height)[:3], (len(all_points), 1))


# Create a visualizer window
vis = o3d.visualization.Visualizer()
vis.create_window()

# Create an initial point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(all_points)


vis.add_geometry(pcd)
###########################

# Add the point cloud to the visualizer
vis.add_geometry(pcd)
vis.update_geometry(pcd)

total_points_threshold=1e6

counts=0
# Function to update the point cloud
while True:
	###Get robot joint data
	res, fb_data = client.fb.try_receive_state_sync(client.controller_info, 0.001)
	if res:
		if fb_data.controller_flags & 0x08 == 0 and counts>1000:
			client.servoMH(False)
			break
		q1_cur=fb_data.group_state[0].feedback_position
		positioner_cur=fb_data.group_state[2].feedback_position
		###get scan data >1% intensity and >50mm in Z
		wire_packet=scan_change.TryGetInValue()
		valid_indices=np.where(wire_packet[1].I_data>1)[0]
		valid_indices=np.intersect1d(valid_indices,np.where(np.abs(wire_packet[1].Z_data)>50)[0])
		line_profile=np.hstack((wire_packet[1].Y_data[valid_indices].reshape(-1,1),wire_packet[1].Z_data[valid_indices].reshape(-1,1)))
		#filter Y within +/10mm
		line_profile=line_profile[np.where(np.abs(line_profile[:,0])<10)]
		#NEW: filter through number of neighbors 2d
		outlier_indices=[]
		for i in range(1,len(line_profile)-1):
			distances=np.linalg.norm(line_profile-line_profile[i],axis=1)
			if np.sum(distances<2)<10:
				outlier_indices.append(i)
		line_profile=np.delete(line_profile,outlier_indices,axis=0)
		#guassian filter
		line_profile = gaussian_filter1d(line_profile, sigma=0.2)
		points_new_cam_frame=np.vstack((np.zeros(len(line_profile)),line_profile.T)).T
		
		cam_pose=robot_fuji.fwd(q1_cur)
		positioner_pose=positioner.fwd(positioner_cur,world=True)
		###transform to positioner's frame
		H_cam=H_from_RT(cam_pose.R,cam_pose.p)
		H_positioner=H_from_RT(positioner_pose.R,positioner_pose.p)
		H_cam2positioner=H_inv(H_positioner)@H_cam
		points_new=(H_cam2positioner[:3,:3]@points_new_cam_frame.T+H_cam2positioner[:3,3].reshape(3,1)).T
		
		###3d filtering based on Z
		valid_indices=np.where(points_new[:,2]>-5)[0]
		points_new=points_new[valid_indices]
		###3d filtering based on distances to original blade
		valid_indices=[]
		for i in range(len(points_new)):
			distances_w_original=np.linalg.norm(curve_sliced_pc-points_new[i],axis=1)
			if np.min(distances_w_original)<20:
				valid_indices.append(i)
			# distances_w_existing=np.linalg.norm(all_points-points_new[i],axis=1)
			# if np.min(distances_w_original)<20 and np.sum(distances_w_existing<5)<20:
			# 	valid_indices.append(i)
		points_new=points_new[valid_indices]

		all_points = np.vstack((np.asarray(pcd.points), points_new)) if counts > 0 else points_new
		all_points_color = np.vstack((np.asarray(pcd.colors), np.tile(cmap(np.mean(points_new[:,2]) / estimated_max_height)[:3], (len(points_new), 1)))) if counts > 0 else np.tile(cmap(np.mean(points_new[:,2]) / estimated_max_height)[:3], (len(points_new), 1))


		counts+=1

	pcd.points = o3d.utility.Vector3dVector(all_points)
	pcd.colors = o3d.utility.Vector3dVector(all_points_color)
	###3d cluster filtering
	# labels = np.array(pcd.cluster_dbscan(eps=0.75, min_points=50, print_progress=False))
	# pcd=pcd.select_by_index(np.argwhere(labels>=0))
	
	###outliers removing
	# cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=100.0)
	# pcd = pcd.select_by_index(ind)

	# Downsample the point cloud if the number of points exceeds the threshold
	# if len(pcd.points) > total_points_threshold:
	# 	pcd = pcd.random_down_sample(total_points_threshold / len(pcd.points))
	
	# Update the visualizer
	vis.update_geometry(pcd)
	vis.poll_events()
	vis.update_renderer()
	
while True:
	try:
		# Update the visualizer
		vis.update_geometry(pcd)
		vis.poll_events()
		vis.update_renderer()
	except:
		break

