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
from dx200_motion_program_exec_client import *
from scipy.interpolate import interp1d
from datetime import datetime
sys.path.append("../../toolbox")
from angled_layers import SpeedHeightModel, LiveFilter, rotate

ir_updated_flag = False
ir_process_packet = None

def ir_process_cb(sub, value, ts):
	global ir_updated_flag, ir_process_packet

	ir_process_packet=copy.deepcopy(value)
	ir_updated_flag=True

if __name__ == '__main__':

    ######## IR VARIABLES #########
    global ir_updated_flag, ir_process_packet

    ######## Welding Parameters ########
    ARCON = False
    RECORDING = True
    ONLINE = True # Used to test without connecting to RR services
    POINT_DISTANCE=0.04
    V_NOMINAL =20
    JOB_OFFSET = 200
    STREAMING_RATE = 125.

    DELAY_CORRECTION = 0.0007

    DATASET = 'bent_tube/'
    SLICED_ALG = 'slice_ER_4043_large_hot/'
    DATA_DIR='../../data/'+DATASET+SLICED_ALG
    with open(DATA_DIR+'slicing.yml', 'r') as file:
        slicing_meta = yaml.safe_load(file)

    ####### CONTROLLER PARAMETERS #######
    V_GAIN = 0.39922 # gradient of model at 3mm/s lower bound

    ######## Create Directories ########
    now = datetime.now()
    recorded_dir = now.strftime(
        "../../../recorded_data/ER4043_bent_tube_large_hot_streaming_%Y_%m_%d_%H_%M_%S/"
    )
    os.makedirs(recorded_dir)

    ######## SENSORS ########
    if ONLINE:
        weld_ser = RRN.SubscribeService('rr+tcp://192.168.55.10:60823?service=welder')
        cam_ser=RRN.ConnectService('rr+tcp://localhost:60827/?service=camera')
        # mic_ser = RRN.ConnectService('rr+tcp://192.168.55.20:60828?service=microphone')

        rr_sensors = WeldRRSensor(weld_service=weld_ser,
                                  cam_service=cam_ser,
                                  microphone_service=None)

    ######## ROBOTS ########
    # Define Kinematics
    CONFIG_DIR = '../../config/'
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
        SS=StreamingSend(RR_robot_sub, streaming_rate=STREAMING_RATE)
        # initialize client to read joint angles
        client = MotionProgramExecClient()
    
    ######## BASE LAYERS ##########

    ######## UPDATE HEIGHT OFFSET IN SEPARATE SCRIPT AND CONNECT TO FLIR #######
    input("Fix height offset, then press enter to continue")
    
    sub=RRN.SubscribeService('rr+tcp://localhost:12182/?service=FLIR_RR_PROCESS')
    ir_process_result=sub.SubscribeWire("ir_process_result")
    # TODO: Not sure what this does, need to fix
    ir_process_result.WireValueChanged += ir_process_cb

    ######## NORMAL LAYERS ########
    num_layer_start = int(0)
    num_layer_end = int(30)
    for layer in range(num_layer_start, num_layer_end):
        ######## INITIALIZE SAVE DIR #######
        save_path = recorded_dir + f"layer_{layer}/"
        os.mkdir(save_path)
        ######## LOAD POINT DATA ########
        rob1_js = np.loadtxt(DATA_DIR+f'curve_sliced_js/MA2010_js{layer}_0.csv',
                             delimiter=',')
        positioner_js = np.loadtxt(DATA_DIR+f'curve_sliced_js/D500B_js{layer}_0.csv', 
                                   delimiter=',')
        curve_sliced_relative = np.loadtxt(DATA_DIR+f'curve_sliced_relative/slice{layer}_0.csv',
                                           delimiter=',')
        curve_sliced= np.loadtxt(DATA_DIR+f'curve_sliced/slice{layer}_0.csv',
                                           delimiter=',')
        lam_relative = calc_lam_cs(curve_sliced_relative)
        print(lam_relative.shape)
        print("------Slice Loaded------")

        # read slicing params
        
        point_of_rotation = np.array(
                (slicing_meta["point_of_rotation"], slicing_meta["baselayer_thickness"])
            )
        base_thickness = slicing_meta["baselayer_thickness"]
        layer_angle = np.array((slicing_meta["layer_angle"]))
        to_flat_angle = np.deg2rad(layer_angle * (layer - 1))
        H = np.loadtxt(data_dir + "curve_pose.csv", delimiter=",")
        p = H[:3, -1]
        R = H[:3, :3]


        # initialize feedrate and velocity
        feedrate=160
        v_cmd = 0

        # calculate velocity profile
        
        dist_to_por = []
        for i in range(len(curve_sliced)):
            point = np.array((curve_sliced[i, 0], curve_sliced[i, 2]))
            dist = np.linalg.norm(point - point_of_rotation)
            dist_to_por.append(dist)

        height_profile = []
        for distance in dist_to_por:
            height_profile.append(distance * np.sin(np.deg2rad(layer_angle)))

        model = SpeedHeightModel(a=-0.36997977, b=1.21532975)
            # model = SpeedHeightModel()
        vel_nom = model.dh2v(height_profile)
        vel_profile = vel_nom

        # Define Start Positions
        MEASURE_DIST = 500 # mm?
        H2010_1440=H_inv(robot2.base_H)
        q_positioner_home=np.array([-15.*np.pi/180.,np.pi/2])
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

        # jog to start position
        input("Press Enter to jog to start position")
        if ONLINE:
            SS.jog2q(np.hstack((rob1_js[0], q2, positioner_js[0])))


        ######## FILTER ########
        filter = LiveFilter()
        error=0
        lam_cur=0
        q_cmd_all = []

        # Looping through the entire path of the sliced part
        input("press enter to start layer")
        if RECORDING:
            rr_sensors.start_all_sensors()
            SS.start_recording()
        if ARCON:
            fronius_client.job_number = int(feedrate/10+JOB_OFFSET)
            fronius_client.start_weld()
        while lam_cur<lam_relative[-1] - v_cmd/STREAMING_RATE:
            loop_start = time.perf_counter()

            # calculate nominal vel of segment
            vel_idx = np.where(lam_relative<=lam_cur)[0][-1]
            v_plan = vel_profile[vel_idx]
            if ir_updated_flag:
                ir_updated_flag=False
                # transform to neutral
                # TODO: See if this brings in more than one image reading
                flame_3d = R.T @ ir_process_packet.flame_position
                # rotate to flat
                new_x, new_z = rotate(
                    point_of_rotation, (flame_3d[0], flame_3d[2]), to_flat_angle
                )
                error = filter.process(new_z - base_thickness)
            # update with P control
            v_cmd = v_plan+V_GAIN*error

            # threshold to stop value from being outside
            v_cmd = max(min(v_cmd, v_upper),v_lower)

            lam_cur += v_cmd/STREAMING_RATE
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

            # log q_cmd
            q_cmd_all.append(np.hstack((time.perf_counter(),q_cmd)))
            # this function has a delay when loop_start is passed in. 
            # Ensures the update frequency is consistent
            # if (loop_start-time.perf_counter())>1/STREAMING_RATE: 
            #     print("Stopping: Loop Time exceeded streaming period")
            #     break
            # input("sending vel command")
            if ONLINE: SS.position_cmd(q_cmd, loop_start+DELAY_CORRECTION) # adding delay to counteract delay in streaming send
        if ARCON:
            fronius_client.stop_weld()
        print(f"-----End of Layer {layer}-----")
        if RECORDING:
            js_recording = SS.stop_recording()
            rr_sensors.stop_all_sensors()
            rr_sensors.save_all_sensors(save_path)
            np.savetxt(save_path+'weld_js_cmd.csv',np.array(q_cmd_all),delimiter=',')
            np.savetxt(save_path+'weld_js_exe.csv',np.array(js_recording),delimiter=',')

        # jog arm1 out of the way
        if ONLINE:
            q_0 = client.getJointAnglesMH(robot.pulse2deg)
            q_0[1] = q_0[1] - np.pi / 8
            SS.jog2q(np.hstack((q_0, q2, positioner_js[-1])))
