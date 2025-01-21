import time, os, copy, yaml
import numpy as np
from motoman_def import robot_obj, positioner_obj
from lambda_calc import calc_lam_cs
from RobotRaconteur.Client import *
from weldRRSensor import *
from dual_robot import *
from traj_manipulation import *
from StreamingSend import StreamingSend
from robotics_utils import H_inv, VectorPlaneProjection
from scipy.interpolate import interp1d


if __name__ == '__main__':
    
    ######## Welding Parameters ########
    ARCON = False
    RECORDING = False
    ONLINE = False # Used to test without connecting to RR services
    POINT_DISTANCE=0.04
    V_NOMINAL = 10
    JOB_OFFSET = 200
    STREAMING_RATE = 125.

    DATASET = 'two_pt_stream_test/'
    SLICED_ALG = 'slice/'
    DATA_DIR='../data/'+DATASET+SLICED_ALG
    with open(DATA_DIR+'slicing.yml', 'r') as file:
        slicing_meta = yaml.safe_load(file)


    ######## SENSORS ########
    if ONLINE:
        # weld_ser = RRN.SubscribeService('rr+tcp://192.168.55.10:60823?service=welder')
        cam_ser=RRN.ConnectService('rr+tcp://localhost:60827/?service=camera')
        # mic_ser = RRN.ConnectService('rr+tcp://192.168.55.20:60828?service=microphone')

        rr_sensors = WeldRRSensor(weld_service=None,cam_service=cam_ser,microphone_service=None)

    ######## FLIR PROCESS ########
    # TODO: Write a service to do the processing I need on the fly with the FLIR
    #       Update this accordingly
    #       Below is how Honglu Implemented it, but my update process is different

    # sub=RRN.SubscribeService('rr+tcp://localhost:12182/?service=FLIR_RR_PROCESS')
    # ir_process_result=sub.SubscribeWire("ir_process_result")
    # TODO: Not sure what this does, need to fix
    # ir_process_result.WireValueChanged += ir_process_cb

    ######## ROBOTS ########
    # Define Kinematics
    CONFIG_DIR = '../config/'
    robot=robot_obj(
        'MA2010_A0',
        def_path=CONFIG_DIR+'MA2010_A0_robot_default_config.yml',
        tool_file_path=CONFIG_DIR+'torch.csv',
        pulse2deg_file_path=CONFIG_DIR+'MA2010_A0_pulse2deg_real.csv',
        d=15
    )
    robot2=robot_obj(
        'MA1440_A0',
        def_path=CONFIG_DIR+'MA1440_A0_robot_default_config.yml',
        tool_file_path=CONFIG_DIR+'flir.csv',
        pulse2deg_file_path=CONFIG_DIR+'MA1440_A0_pulse2deg_real.csv',
        base_transformation_file=CONFIG_DIR+'MA1440_pose.csv'
    )
    positioner=positioner_obj(
        'D500B',
        def_path=CONFIG_DIR+'D500B_robot_extended_config.yml',
        tool_file_path=CONFIG_DIR+'positioner_tcp.csv',
        pulse2deg_file_path=CONFIG_DIR+'D500B_pulse2deg_real.csv',
        base_transformation_file=CONFIG_DIR+'D500B_pose.csv'
    )

    # Define Start Positions
    MEASURE_DIST = 500 # mm?
    H2010_1440=H_inv(robot2.base_H)
    q_positioner_home=np.array([-15.*np.pi/180.,np.pi/2])
    rob1_js=np.loadtxt(DATA_DIR+'curve_sliced_js/MA2010_js0_0.csv',delimiter=',')
    positioner_js=np.loadtxt(DATA_DIR+'curve_sliced_js/D500B_js0_0.csv', delimiter=',')
    p_positioner_home=np.mean([robot.fwd(rob1_js[0]).p, robot.fwd(rob1_js[-1]).p], axis=0)
    p_robot2_proj=p_positioner_home+np.array([0,0,50])
    p2_in_base_frame=np.dot(H2010_1440[:3,:3],p_robot2_proj)+H2010_1440[:3,3]

	###pointing toward positioner's X with 15deg tilted angle looking down
    v_z=H2010_1440[:3,:3]@np.array([0,-0.96592582628,-0.2588190451])

    ###FLIR's Y pointing toward 1440's -X in 1440's base frame, projected on v_z's plane
    v_y=VectorPlaneProjection(np.array([-1,0,0]),v_z)
    v_x=np.cross(v_y,v_z)

    ###back project measure_distance-mm away from torch
    p2_in_base_frame=p2_in_base_frame-MEASURE_DIST*v_z
    R2=np.vstack((v_x,v_y,v_z)).T
    q2=robot2.inv(p2_in_base_frame,R2,last_joints=np.zeros(6))[0]


    ######## RR FRONIUS ########
    if ARCON:
        fronius_sub=RRN.SubscribeService('rr+tcp://192.168.55.21:60823?service=welder')
        fronius_client = fronius_sub.GetDefaultClientWait(1)      #connect, timeout=30s
        hflags_const = RRN.GetConstants(
            "experimental.fronius", 
            fronius_client
        )["WelderStateHighFlags"]
        fronius_client.prepare_welder()


    ######## RR STREAMING ########
    if ONLINE:
        RR_robot_sub = RRN.SubscribeService('rr+tcp://localhost:59945?service=robot')
        SS=StreamingSend(RR_robot_sub, streamingrate=STREAMING_RATE)


    ######## LOAD POINT DATA ########
    # I have already generated one continuous spiral, just need to import points for each robot
    rob1_js = np.loadtxt(DATA_DIR+'curve_sliced_js/MA2010_js1_0.csv', delimiter=',')
    positioner_js = np.loadtxt(DATA_DIR+'curve_sliced_js/D500B_js1_0.csv', delimiter=',')
    curve_sliced_relative = np.loadtxt(DATA_DIR+'curve_sliced_relative/slice1_0.csv', delimiter=',')
    lam_relative = calc_lam_cs(curve_sliced_relative)
    print("------Slices Loaded------")

    ######## WELD CONTINUOUSLY ########

    # initialize feedrate and velocity
    feedrate=160
    v_cmd = 3

    # jog to start position
    if ONLINE: SS.jog2q(np.hstack((rob1_js[0], q2, positioner_js[0])))

    if RECORDING:
        rr_sensors.start_all_sensors()
        SS.start_recording()
    if ARCON:
        fronius_client.job_number = int(feedrate/10+JOB_OFFSET)
        fronius_client.start_weld()

    lam_cur=0

    # Looping through the entire path of the sliced part
    while lam_cur<lam_relative[-1] - v_cmd/STREAMING_RATE:
        loop_start = time.perf_counter()

        lam_cur += v_cmd/STREAMING_RATE
        print(lam_cur)
        # get closest lambda that is greater than current lambda
        lam_idx = np.where(lam_relative>=lam_cur)[0][0]
        # Calculate the fraction of the current lambda that has been traversed
        lam_ratio = ((lam_cur-lam_relative[lam_idx-1])/
                     (lam_relative[lam_idx]-lam_relative[lam_idx-1]))
        # Apply that fraction to the joint space
        q1 = rob1_js[lam_idx-1]*(1-lam_ratio)+rob1_js[lam_idx]*lam_ratio
        q_positioner = positioner_js[lam_idx-1]*(1-lam_ratio) + positioner_js[lam_idx]*(lam_ratio)

        #generate set of joints to command
        q_cmd = np.hstack((q1, q2, q_positioner))
        # TODO: Update IR Images

        # TODO: Calculate Control Inputs (v_T, v_w)

        # TODO: Update Welding Commands

        # this function has a delay when loop_start is passed in. Ensures the update frequency is consistent
        if (loop_start-time.perf_counter())>1/STREAMING_RATE: 
            print("Stopping: Loop Time exceeded streaming period")
            break
        if ONLINE: SS.position_cmd(q_cmd, loop_start)
    print("-----End of Job-----")
