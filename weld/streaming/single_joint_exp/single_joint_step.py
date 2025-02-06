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
    
    ######## Parameters ########
    V_NOMINAL = 0.01 # rad/s
    JOINT_NUM = 1
    ANGLE_START = 0 # rad
    ANGLE_END = -20*np.pi/180 # rad
    RECORDING = False
    ONLINE = True # Used to test without connecting to RR services
    STREAMING_RATE = 125.
    STEP_DIST = 0.01 # rad

    CORRECTION_DELAY = 0.0007

    ######## Create Directories ########
    recorded_dir=f'../../../../recorded_data/streaming_delay/single_joint_{JOINT_NUM}_vel_{V_NOMINAL}_end_{ANGLE_END*180/np.pi}/'
    os.makedirs(recorded_dir, exist_ok=True)

    ######## ROBOTS ########
    # Define Kinematics
    CONFIG_DIR = '../../../config/'
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

    ######## RR STREAMING ########
    if ONLINE:
        RR_robot_sub = RRN.SubscribeService('rr+tcp://localhost:59945?service=robot')
        SS=StreamingSend(RR_robot_sub, streaming_rate=STREAMING_RATE)


    v_cmd = V_NOMINAL

    # jog to start position
    input("Press Enter to jog to start position")
    q1_home = np.radians([0,0,0,0,0,0])
    q2_home = np.radians([90,0,0,0,0,0])
    pos_home = np.radians([-15,0])
    if ONLINE: SS.jog2q(np.hstack((q1_home,q2_home,pos_home))) # should be home, double check

    if RECORDING:
        rr_sensors.start_all_sensors()
        SS.start_recording()

    joint_cur=0
    q2 = q2_home
    q_positioner = pos_home
    q_cmd_all = []

    # Looping through the entire path of the sliced part
    input("press enter to start streaming")
    SS.start_recording()
    while np.abs(joint_cur)<np.abs(ANGLE_END - v_cmd/STREAMING_RATE):
        loop_start = time.perf_counter()

        joint_cur += v_cmd/STREAMING_RATE

        q1 = np.zeros(6)
        q1[JOINT_NUM] = joint_cur

        #generate set of joints to command
        q_cmd = np.hstack((q1, q2, q_positioner))

        # log q_cmd
        q_cmd_all.append(np.hstack((time.perf_counter(),q_cmd)))

        # print("Commanding jointset: ", q_cmd)
        # input("sending vel command")
        if ONLINE:
            SS.position_cmd(q_cmd, loop_start+CORRECTION_DELAY) # adding delay to counteract delay in streaming send 

    print("-----End of Job-----")
    js_recording = SS.stop_recording()
    np.savetxt(recorded_dir+'weld_js_cmd.csv',np.array(q_cmd_all),delimiter=',')
    np.savetxt(recorded_dir+'weld_js_exe.csv',np.array(js_recording),delimiter=',')
