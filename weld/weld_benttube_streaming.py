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
    # TODO: Continue working through honglu's "weld_cylinder_spiral_ir_feedback.py" example
