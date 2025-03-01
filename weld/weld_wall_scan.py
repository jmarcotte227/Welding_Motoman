from copy import deepcopy
from pathlib import Path
import pickle, sys, time, datetime, traceback, glob
sys.path.append('../toolbox/')
sys.path.append('../scan/scan_tools/')
sys.path.append('../scan/scan_plan/')
sys.path.append('../scan/scan_process/')
sys.path.append('../sensor_fusion/')

from robot_def import *
from scan_utils import *
from utils import *
from scanPathGen import *
from scanProcess import *
from weldRRSensor import *
from WeldSend import *
from dx200_motion_program_exec_client import *
from general_robotics_toolbox import *
from RobotRaconteur.Client import *
import matplotlib.pyplot as plt
import numpy as np

def robot_weld_path_gen(all_layer_z,forward_flag,base_layer):
    R=np.array([[-0.7071, 0.7071, -0.    ],
            [ 0.7071, 0.7071,  0.    ],
            [0.,      0.,     -1.    ]])
    x0 =  1684	# Origin x coordinate
    y0 = -1179 + 428	# Origin y coordinate
    z0 = -260   # 10 mm distance to base

    weld_p=[]
    if base_layer: # base layer
        weld_p.append([x0 - 33, y0 - 20, z0+10])
        weld_p.append([x0 - 33, y0 - 20, z0])
        weld_p.append([x0 - 33, y0 - 105 , z0])
        weld_p.append([x0 - 33, y0 - 105 , z0+10])
    else: # top layer
        weld_p.append([x0 - 33, y0 - 30, z0+10])
        weld_p.append([x0 - 33, y0 - 30, z0])
        weld_p.append([x0 - 33, y0 - 95 , z0])
        weld_p.append([x0 - 33, y0 - 95 , z0+10])

    if not forward_flag:
        weld_p = weld_p[::-1]

    all_path_T=[]
    for layer_z in all_layer_z:
        path_T=[]
        for p in weld_p:
            path_T.append(Transform(R,p+np.array([0,0,layer_z])))

        all_path_T.append(path_T)
    
    return all_path_T


zero_config=np.zeros(6)
# 0. robots.
config_dir='../config/'

robot_weld=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',d=15,tool_file_path=config_dir+'torch.csv', pulse2deg_file_path='../config/MA2010_A0_pulse2deg_real.csv')
robot_scan=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'mti.csv',base_transformation_file='../config/MA1440_pose.csv',pulse2deg_file_path='../config/MA1440_A0_pulse2deg_real.csv')
robot_ir=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'flir.csv',base_transformation_file='../config/MA1440_pose.csv',pulse2deg_file_path='../config/MA1440_A0_pulse2deg_real.csv')

positioner=positioner_obj('D500B',def_path=config_dir+'D500B_robot_default_config.yml',tool_file_path=config_dir+'positioner_tcp.csv',base_transformation_file='../config/D500B_pose.csv',pulse2deg_file_path='../config/D500B_pulse2deg_real.csv')


Table_home_T = positioner.fwd(np.radians([-15,180]))
T_S1TCP_R1Base = np.linalg.inv(np.matmul(positioner.base_H,H_from_RT(Table_home_T.R,Table_home_T.p)))
T_R1Base_S1TCP = np.linalg.inv(T_S1TCP_R1Base)


final_height=50
final_h_std_thres=999999999
weld_z_height=[0,1,2] # two base layer height to first top layer
layer_height=0.8
weld_z_height=np.append(weld_z_height,np.linspace(weld_z_height[-1]+layer_height,final_height,int((final_height-weld_z_height[-1])/layer_height)))
feedrate = 150
baselayer_feedrate = 300
job_offset=200

job_number=np.append(2*[int(baselayer_feedrate/10)+job_offset],np.ones(len(weld_z_height)-2)*(int(feedrate/10) + job_offset))


print(weld_z_height)
print(job_number)

ipm_mode=300

v_baselayer=5
weld_v=10

weld_velocity=[v_baselayer]*2+[weld_v]*len(weld_z_height[2:])


to_start_speed=5
to_home_speed=6

save_weld_record=True


current_time = datetime.datetime.now()
formatted_time = current_time.strftime('%Y_%m_%d_%H_%M_%S.%f')[:-7]
data_dir=f'../../recorded_data/wall_weld_test/4043_{feedrate}ipm_'  +formatted_time+'/'


## rr drivers and all other drivers
client=MotionProgramExecClient()
ws=WeldSend(client)
# weld state logging
# current_ser=RRN.SubscribeService('rr+tcp://192.168.55.21:12182?service=Current')
# weld_ser = RRN.SubscribeService('rr+tcp://192.168.55.10:60823?service=welder')
cam_ser= RRN.ConnectService('rr+tcp://localhost:60827/?service=camera')
# mic_ser = RRN.ConnectService('rr+tcp://localhost:60828?service=microphone')
## RR sensor objects
# rr_sensors = WeldRRSensor(weld_service=weld_ser,cam_service=cam_ser,microphone_service=mic_ser,current_service=current_ser)

rr_sensors = WeldRRSensor(cam_service=cam_ser)

# ## test sensor (camera, microphone)
# exit()
###############

# MTI connect to RR
mti_client = RRN.ConnectService("rr+tcp://192.168.55.10:60830/?service=MTI2D")
mti_client.setExposureTime("25")
###################################
base_layer = True
profile_height=None
# Transz0_H=None
Transz0_H=np.array([[ 9.99997540e-01,  2.06703673e-06, -2.21825071e-03, -3.46701381e-03],
 [ 2.06703673e-06,  9.99998263e-01,  1.86365986e-03,  2.91280622e-03],
 [ 2.21825071e-03, -1.86365986e-03,  9.99995803e-01,  1.56294293e+00],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
curve_sliced_relative=None
last_mean_h = 0

# ir pose
###define start pose for 3 robtos
measure_distance=500
H2010_1440=H_inv(robot_ir.base_H)
q_positioner_home=np.array([-15.*np.pi/180.,np.pi])
p_positioner_home=positioner.fwd(q_positioner_home,world=True).p
p_robot2_proj=p_positioner_home+np.array([0,0,50])
p2_in_base_frame=np.dot(H2010_1440[:3,:3],p_robot2_proj)+H2010_1440[:3,3]
v_z=H2010_1440[:3,:3]@np.array([-0.9659258262,0,-0.2588190451]) #R(15)deg down
v_y=VectorPlaneProjection(np.array([0,1,0]),v_z)	###FLIR's Y pointing toward 1440's +Y in 1440's base frame, projected on v_z's plane
v_x=np.cross(v_y,v_z)
p2_in_base_frame=p2_in_base_frame-measure_distance*v_z			###back project measure_distance-mm away from torch
R2=np.vstack((v_x,v_y,v_z)).T
r2_ir_q=robot_ir.inv(p2_in_base_frame,R2,last_joints=np.zeros(6))[0]
r2_mid = np.radians([43.7851,20,-10,0,0,0])

weld_arcon=True

# move robot to ready position
ws.jog_dual(robot_scan,positioner,r2_mid,np.radians([-15,180]),to_start_speed)
ws.jog_dual(robot_scan,positioner,r2_ir_q,np.radians([-15,180]),to_start_speed)

for i in range(0,len(weld_z_height)):
    cycle_st = time.time()
    print("==================================")
    print("Layer:",i)
    if i%2==0:
        forward_flag = True
    else:
        forward_flag = False
    #### welding
    weld_st = time.time()
    if i>=0 and True:
        weld_plan_st = time.time()
        if i>=2:
            base_layer=False
        this_z_height=weld_z_height[i]
        this_job_number=job_number[i]
        this_weld_v=[weld_velocity[i]]
        all_dh=[]

        # 1. The curve path in "positioner tcp frame"
        ######### enter your wanted z height #######
        all_layer_z = [this_z_height]
        ###########################################
        all_path_T = robot_weld_path_gen(all_layer_z,forward_flag,base_layer) # this is your path
        path_T=all_path_T[0]
        curve_sliced_relative=[]
        for path_p in path_T:
            this_p = np.matmul(T_S1TCP_R1Base[:3,:3],path_p.p)+T_S1TCP_R1Base[:3,-1]
            this_n = np.matmul(T_S1TCP_R1Base[:3,:3],path_p.R[:,-1])
            curve_sliced_relative.append(np.append(this_p,this_n))
        curve_sliced_relative=curve_sliced_relative[1:-1] # the start and end is for collision prevention
        print(path_T[0])
        print(curve_sliced_relative)
        R_S1TCP = np.matmul(T_S1TCP_R1Base[:3,:3],path_p.R)

        h_largest=this_z_height
        if (last_mean_h == 0) and (profile_height is not None):
            last_mean_h=np.mean(profile_height[:,1])
        

        if (profile_height is not None) and (i>2):
            mean_h = np.mean(profile_height[:,1])
            dh_last_layer = mean_h-last_mean_h
            h_target = mean_h+dh_last_layer

            dh_direction = np.array([0,0,h_target-curve_sliced_relative[0][2]])
            dh_direction_R1 = T_R1Base_S1TCP[:3,:3]@dh_direction

            for curve_i in range(len(curve_sliced_relative)):
                curve_sliced_relative[curve_i][2]=h_target
            
            for path_i in range(len(path_T)):
                path_T[path_i].p=path_T[path_i].p+dh_direction_R1

            last_mean_h=mean_h
                

        
        path_q = []
        for tcp_T in path_T:
            path_q.append(robot_weld.inv(tcp_T.p,tcp_T.R,zero_config)[0])

        
        ####################
        print("dh:",all_dh)
        print("Nominal V:",weld_velocity[i])
        print("Correct V:",this_weld_v)
        print("curve_sliced_relative:",curve_sliced_relative)
        print(path_T[0])
        print(len(path_T))
        print(len(curve_sliced_relative))

        print("Weld Plan time:",time.time()-weld_plan_st)

        ######################################################
        ########### Do welding #############
        
        # input("Press Enter and move to weld starting point.")
        ws.jog_single(robot_weld,path_q[0],to_start_speed)
        
        weld_motion_weld_st = time.time()

        primitives=[]
        for bpi in range(len(this_weld_v)+1):
            primitives.append('movel')

        rr_sensors.start_all_sensors()


        ws.weld_segment_single(primitives,robot_weld,path_q[1:-1],np.append(10,this_weld_v),cond_all=[int(this_job_number)],arc=weld_arcon,blocking=False)
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
                    js_recording.append(np.array([time.time()]+[fb_data.job_state[0][1]]+client.joint_angle.tolist()))
        rr_sensors.stop_all_sensors()
        client.servoMH(False) #stop the motor

        ##############################################################Log Sensor Data####################################################################
        rr_sensors.stop_all_sensors()

        if save_weld_record:
            os.makedirs(data_dir, exist_ok=True)
            layer_data_dir=data_dir+'layer_'+str(i)+'/'
            Path(layer_data_dir).mkdir(exist_ok=True)
            # save cmd
            q_bp=[]
            for q in path_q[1:-1]:
                q_bp.append([np.array(q)])
            # save weld record
            np.savetxt(layer_data_dir + 'weld_js_exe.csv',np.array(js_recording),delimiter=',')
            np.save(layer_data_dir+'primitives.npy',primitives)
            np.save(layer_data_dir+'path_q.npy',path_q)
            np.save(layer_data_dir+'this_weld_v.npy',np.append(10,this_weld_v))
            try:
                rr_sensors.save_all_sensors(layer_data_dir)
            except:
                traceback.print_exc()
        
        print("Weld actual weld time:",time.time()-weld_motion_weld_st)
        weld_to_home_st = time.time()

        # move home
        # input("Press Enter to Move Home")
        print("Weld to home time:",time.time()-weld_to_home_st)
        ######################################################

        print("Weld Time:",time.time()-weld_st)

    ws.jog_single(robot_weld,np.zeros(6),to_home_speed)


    #### scanning
    scan_st = time.time()
    scan_plan_st = time.time()
    # 2. Scanning parameters
    ### scan parameters
    scan_speed=10 # scanning speed (mm/sec)
    scan_stand_off_d = 95 ## mm
    Rz_angle = np.radians(0) # point direction w.r.t welds
    Ry_angle = np.radians(0) # rotate in y a bit
    bounds_theta = np.radians(1) ## circular motion at start and end
    all_scan_angle = np.radians([0]) ## scan angle
    q_init_table=np.radians([-15,200]) ## init table
    save_output_points = True
    ### scanning path module
    spg = ScanPathGen(robot_scan,positioner,scan_stand_off_d,Rz_angle,Ry_angle,bounds_theta)
    mti_Rpath = np.array([[ -1.,0.,0.],   
                [ 0.,1.,0.],
                [0.,0.,-1.]])
    # generate scan path
    if forward_flag:
        scan_p,scan_R,q_out1,q_out2=spg.gen_scan_path([curve_sliced_relative],[0],all_scan_angle,\
                        solve_js_method=0,q_init_table=q_init_table,R_path=mti_Rpath,scan_path_dir=None)
    else:
        scan_p,scan_R,q_out1,q_out2=spg.gen_scan_path([curve_sliced_relative[::-1]],[0],all_scan_angle,\
                        solve_js_method=0,q_init_table=q_init_table,R_path=mti_Rpath,scan_path_dir=None)
    # generate motion program
    q_bp1,q_bp2,s1_all,s2_all=spg.gen_motion_program(q_out1,q_out2,scan_p,scan_speed,init_sync_move=0)
    #######################################

    print("Scan plan time:",time.time()-scan_plan_st)

    scan_motion_st = time.time()
    ######## scanning motion #########

    ## move to start
    ws.jog_dual(robot_scan,positioner,r2_mid,q_bp2[0][0],to_start_speed)
    ws.jog_dual(robot_scan,positioner,q_bp1[0][0],q_bp2[0][0],to_start_speed)

    # input("Press Enter to start moving and scanning")
    scan_motion_scan_st = time.time()

    ## motion start
    mp = MotionProgram(ROBOT_CHOICE='RB2',ROBOT_CHOICE2='ST1',pulse2deg=robot_scan.pulse2deg,pulse2deg_2=positioner.pulse2deg)
    # calibration motion
    target2=['MOVJ',np.degrees(q_bp2[1][0]),s2_all[0]]
    mp.MoveL(np.degrees(q_bp1[1][0]), scan_speed, 0, target2=target2)
    # routine motion
    for path_i in range(2,len(q_bp1)-1):
        target2=['MOVJ',np.degrees(q_bp2[path_i][0]),s2_all[path_i]]
        mp.MoveL(np.degrees(q_bp1[path_i][0]), s1_all[path_i], target2=target2)
    target2=['MOVJ',np.degrees(q_bp2[-1][0]),s2_all[-1]]
    mp.MoveL(np.degrees(q_bp1[-1][0]), s1_all[-1], 0, target2=target2)

    ws.client.execute_motion_program_nonblocking(mp)
    ###streaming
    ws.client.StartStreaming()
    start_time=time.time()
    state_flag=0
    joint_recording=[]
    robot_stamps=[]
    mti_recording=[]
    r_pulse2deg = np.append(robot_scan.pulse2deg,positioner.pulse2deg)
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
            mti_recording.append(deepcopy(np.array([mti_client.lineProfile.X_data,mti_client.lineProfile.Z_data])))
    ws.client.servoMH(False)
    
    mti_recording=np.array(mti_recording)
    joint_recording=np.array(joint_recording)
    q_out_exe=joint_recording[:,6:]

    q2=deepcopy(r2_ir_q)
    q3=np.radians([-15,180])
    ws.jog_dual(robot_scan,positioner,r2_mid,q3,to_home_speed)
    ws.jog_dual(robot_scan,positioner,r2_ir_q,q3,to_home_speed)
    #####################

    print("Total exe len:",len(q_out_exe))
    if save_output_points:
        out_scan_dir = layer_data_dir+'scans/'
        ## save traj
        Path(out_scan_dir).mkdir(exist_ok=True)
        # save poses
        np.savetxt(out_scan_dir + 'scan_js_exe.csv',q_out_exe,delimiter=',')
        np.savetxt(out_scan_dir + 'scan_robot_stamps.csv',robot_stamps,delimiter=',')
        with open(out_scan_dir + 'mti_scans.pickle', 'wb') as file:
            pickle.dump(mti_recording, file)
        print('Total scans:',len(mti_recording))
    
    print("Scan motion time:",time.time()-scan_motion_st)
    
    print("Scan Time:",time.time()-scan_st)


    ########################

    recon_3d_st = time.time()
    #### scanning process: processing point cloud and get h
    try:
        if forward_flag:
            curve_x_start = deepcopy(curve_sliced_relative[0][0])
            curve_x_end = deepcopy(curve_sliced_relative[-1][0])
        else:
            curve_x_start = deepcopy(curve_sliced_relative[-1][0])
            curve_x_end = deepcopy(curve_sliced_relative[0][0])
    except:
        curve_x_start=43
        curve_x_end=-41

    z_height_start=h_largest-3
    crop_extend=10
    crop_min=(curve_x_end-crop_extend,-30,-10)
    crop_max=(curve_x_start+crop_extend,30,z_height_start+30)
    crop_h_min=(curve_x_end-crop_extend,-20,-10)
    crop_h_max=(curve_x_start+crop_extend,20,z_height_start+30)
    scan_process = ScanProcess(robot_scan,positioner)
    pcd=None
    pcd = scan_process.pcd_register_mti(mti_recording,q_out_exe,robot_stamps,static_positioner_q=q_init_table)
    pcd = scan_process.pcd_noise_remove(pcd,nb_neighbors=40,std_ratio=1.5,\
                                        min_bound=crop_min,max_bound=crop_max,cluster_based_outlier_remove=True,cluster_neighbor=1,min_points=100)
    print("3D Reconstruction:",time.time()-recon_3d_st)
    get_h_st = time.time()
    profile_height,Transz0_H = scan_process.pcd2height(deepcopy(pcd),z_height_start,bbox_min=crop_h_min,bbox_max=crop_h_max,Transz0_H=Transz0_H)
    print("Transz0_H:",Transz0_H)
    
    print("Get Height:",time.time()-get_h_st)

    save_output_points=True
    if save_output_points:
        o3d.io.write_point_cloud(out_scan_dir+'processed_pcd.pcd',pcd)
        np.save(out_scan_dir+'height_profile.npy',profile_height)
    # visualize_pcd([pcd])
    plt.scatter(profile_height[:,0],profile_height[:,1])


    if np.mean(profile_height[:,1])>final_height and np.std(profile_height[:,1])<final_h_std_thres:
        break

    forward_flag=not forward_flag

    print("Print Cycle Time:",time.time()-cycle_st)

print("Welding End!!")