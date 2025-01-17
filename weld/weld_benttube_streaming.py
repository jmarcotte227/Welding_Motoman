import time, os, copy
from motoman_def import *
from lambda_calc import *
from RobotRaconteur.Client import *
from weldRRSensor import *
from dual_robot import *
from traj_manipulation import *
from StreamingSend import *


if __name__ == '__main__':

    ARCON = False

    DATASET = 'bent_tube_continuous/'
    SLICED_ALG = 'slice_ER_4043/'
    DATA_DIR='../data/'+DATASET+SLICED_ALG
    with open(DATA_DIR+'slicing.yaml', 'r') as file:
        slicing_meta = yaml.safe_load(file)


    ######## SENSORS ########
    # weld_ser = RRN.SubscribeService('rr+tcp://192.168.55.10:60823?service=welder')
    cam_ser=RRN.ConnectService('rr+tcp://localhost:60827/?service=camera')
	# mic_ser = RRN.ConnectService('rr+tcp://192.168.55.20:60828?service=microphone')

    rr_sensors = WeldRRSensor(weld_service=None,cam_service=cam_ser,microphone_service=None)

    ######## FLIR PROCESS ########
    # TODO: Write a service to do the processing I need on the fly with the FLIR
    #       Update this accordingly

    sub=RRN.SubscribeService('rr+tcp://localhost:12182/?service=FLIR_RR_PROCESS')
    ir_process_result=sub.SubscribeWire("ir_process_result")
    # TODO: Not sure what this does, need to fix
    ir_process_result.WireValueChanged += ir_process_cb

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
        base_transformation_file=CONFIG_DIR+'D500B_pose_mocap.csv'
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

	###pointing toward positioner's X with 15deg tiltd angle looking down
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
    RR_robot_sub = RRN.SubscribeService('rr+tcp://localhost:59945?service=robot')
    POINT_DISTANCE=0.04
    SS=StreamingSend(RR_robot_sub, streamingrate=125.)

    # TODO: FINISH THIS
