import time, os, copy, yaml
from datetime import datetime
import numpy as np
from numpy.linalg import norm
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
import torch
sys.path.append("../../toolbox")
from angled_layers import SpeedHeightModel, rotate, flame_tracking_stream, delta_v, avg_by_line, interpolate_heights, LiveAverageFilter, calc_velocity_stream
from lstm_model_next_step_fast import WeldLSTM

ir_updated_flag = False
ir_process_packet = None
ir_process_output = []

# Initialize Filters
pos_filter = LiveAverageFilterPos()
temp_filter = LiveAverageFilterScalar()


def ir_process_cb(sub, value, ts):
    global ir_updated_flag, ir_process_packet, ir_process_output, pos_obj, temp_obj
    ir_process_packet=copy.deepcopy(value)
    ir_process_output.append(ir_process_packet.flame_position)
    # add z position to averaging filter
    pos_filter.log_reading(ir_process_packet.flame_position)
    temp_filter.log_reading(ir_process_packet.avg_flame_temp)
    ir_updated_flag=True

def main():

    ######## IR VARIABLES #########
    global ir_updated_flag, ir_process_packet, ir_process_output, pos_filter, temp_filter

    ######## Welding Parameters ########
    ARCON = False
    RECORDING = True
    ONLINE = True # Used to test without connecting to RR services
    V_NOMINAL =20
    BASE_VEL = 3
    BASE_FEEDRATE= 300
    FEEDRATE = 160
    JOB_OFFSET = 200
    STREAMING_RATE = 125.

    DELAY_CORRECTION = 0.0007

    DATASET = 'bent_tube/'
    SLICED_ALG = 'slice_ER_4043_large_hot/'
    DATA_DIR='../../data/'+DATASET+SLICED_ALG
    with open(DATA_DIR+'slicing.yml', 'r') as file:
        slicing_meta = yaml.safe_load(file)

    ####### CONTROLLER PARAMETERS #######
    # V_GAIN = 3.612# changed from this at layer 72 700.39922 # gradient of model at 3mm/s lower bound
    V_LOWER = 3 # mm/s
    V_UPPER = 17 # mm/s

    ######## Create Directories ########
    now = datetime.now()
    recorded_dir = now.strftime(
        "../../../recorded_data/ER4043_bent_tube_large_hot_streaming_%Y_%m_%d_%H_%M_%S/"
    )
    os.makedirs(recorded_dir)
    # recorded_dir = "../../../recorded_data/ER4043_bent_tube_large_hot_streaming_2025_03_12_09_32_31/"

    ######## SENSORS ########
    if ONLINE:
        # weld_ser = RRN.SubscribeService('rr+tcp://192.168.55.10:60823?service=welder')
        cam_ser=RRN.ConnectService('rr+tcp://localhost:60827/?service=camera')
        # mic_ser = RRN.ConnectService('rr+tcp://192.168.55.20:60828?service=microphone')

        rr_sensors = WeldRRSensor(weld_service=None,
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
    flir_intrinsic = yaml.load(open(CONFIG_DIR + "FLIR_A320.yaml"), Loader=yaml.FullLoader)
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
    #layer = 0
    #save_path = recorded_dir + f"layer_{layer}/"
    #os.mkdir(save_path)

    ######### LOAD POINT DATA ########
    #rob1_js = np.loadtxt(DATA_DIR+f'curve_sliced_js/MA2010_js{layer}_0.csv',
    #                     delimiter=',')
    #positioner_js = np.loadtxt(DATA_DIR+f'curve_sliced_js/D500B_js{layer}_0.csv',
    #                           delimiter=',')
    #curve_sliced_relative = np.loadtxt(DATA_DIR+f'curve_sliced_relative/slice{layer}_0.csv',
    #                                   delimiter=',')
    #curve_sliced= np.loadtxt(DATA_DIR+f'curve_sliced/slice{layer}_0.csv',
    #                                   delimiter=',')
    #lam_relative = calc_lam_cs(curve_sliced_relative)
    #H = np.loadtxt(DATA_DIR + "curve_pose.csv", delimiter=",")
    #p = H[:3, -1]
    #R = H[:3, :3]
    #print(lam_relative.shape)
    #print("------Slice Loaded------")
    ## Define Start Positions
    #MEASURE_DIST = 500 # mm?
    #H2010_1440=H_inv(robot2.base_H)
    #q_positioner_home=np.array([-15.*np.pi/180.,np.pi/2])
    #p_positioner_home=np.mean([robot.fwd(rob1_js[0]).p, robot.fwd(rob1_js[-1]).p], axis=0)
    #p_robot2_proj=p_positioner_home+np.array([0,0,50])
    #p2_in_base_frame=np.dot(H2010_1440[:3,:3],p_robot2_proj)+H2010_1440[:3,3]

    ####pointing toward positioner's X with 15deg tilted angle looking down
    #v_z=H2010_1440[:3,:3]@np.array([0,-0.96592582628,-0.2588190451])

    ####FLIR's Y pointing toward 1440's -X in 1440's base frame, projected on v_z's plane
    #v_y=VectorPlaneProjection(np.array([-1,0,0]),v_z)
    #v_x=np.cross(v_y,v_z)

    ####back project measure_distance-mm away from torch
    #p2_in_base_frame=p2_in_base_frame-MEASURE_DIST*v_z
    #R2=np.vstack((v_x,v_y,v_z)).T
    #q2=robot2.inv(p2_in_base_frame,R2,last_joints=np.zeros(6))[0]

    ## jog to start position
    #input("Press Enter to jog to start position")
    #if ONLINE:
    #    SS.jog2q(np.hstack((rob1_js[0], q2, positioner_js[0])))

    #error=0
    #lam_cur=0
    #q_cmd_all = []
    #job_no = []
    #v_cmd = BASE_VEL

    #input("press enter to start layer")
    #if RECORDING:
    #    rr_sensors.start_all_sensors()
    #    SS.start_recording()
    #if ARCON:
    #    fronius_client.job_number = int(BASE_FEEDRATE/10+JOB_OFFSET)
    #    fronius_client.start_weld()
    #while lam_cur<lam_relative[-1] - v_cmd/STREAMING_RATE:
    #    loop_start = time.perf_counter()

    #    # calculate nominal vel of segment
    #    vel_idx = np.where(lam_relative<=lam_cur)[0][-1]

    #    lam_cur += v_cmd/STREAMING_RATE
    #    # get closest lambda that is greater than current lambda
    #    lam_idx = np.where(lam_relative>=lam_cur)[0][0]
    #    # Calculate the fraction of the current lambda that has been traversed
    #    lam_ratio = ((lam_cur-lam_relative[lam_idx-1])/
    #                 (lam_relative[lam_idx]-lam_relative[lam_idx-1]))
    #    # Apply that fraction to the joint space
    #    q1 = rob1_js[lam_idx-1]*(1-lam_ratio)+rob1_js[lam_idx]*lam_ratio
    #    q_positioner = positioner_js[lam_idx-1]*(1-lam_ratio) + positioner_js[lam_idx]*(lam_ratio)

    #    #generate set of joints to command
    #    q_cmd = np.hstack((q1, q2, q_positioner))

    #    # log q_cmd
    #    q_cmd_all.append(np.hstack((time.perf_counter(),q_cmd)))
    #    job_no.append(vel_idx)

    #    # this function has a delay when loop_start is passed in.
    #    # Ensures the update frequency is consistent
    #    # if (loop_start-time.perf_counter())>1/STREAMING_RATE:
    #    #     print("Stopping: Loop Time exceeded streaming period")
    #    #     break

    #    # adding delay to counteract delay in streaming send
    #    if ONLINE: 
    #        SS.position_cmd(q_cmd, loop_start+DELAY_CORRECTION) 
    #if ARCON:
    #    fronius_client.stop_weld()
    #print(f"-----End of Layer {layer}-----")
    #if RECORDING:
    #    js_recording = SS.stop_recording()
    #    rr_sensors.stop_all_sensors()
    #    rr_sensors.save_all_sensors(save_path)
    #    job_no = np.reshape(np.array(job_no), (-1,1))
    #    q_cmd_all = np.array(q_cmd_all)
    #    js_recording = np.array(js_recording)
    #    cmd_out = np.hstack((
    #        np.reshape(q_cmd_all[:,0],(-1,1)),
    #        job_no,
    #        q_cmd_all[:,1:]
    #        ))
    #    job_no_exe = []
    #    # find closest jobno for exe
    #    for t in js_recording[:,0]:
    #        idx = np.argmin(np.abs(cmd_out[:,0]-t))
    #        job_no_exe.append(cmd_out[idx,1])

    #    job_no_exe = np.reshape(np.array(job_no_exe), (-1,1))
    #    exe_out = np.hstack((
    #        np.reshape(js_recording[:,0],(-1,1)),
    #        job_no_exe,
    #        js_recording[:,1:]
    #        ))
    #    np.savetxt(save_path+'weld_js_cmd.csv',cmd_out,delimiter=',')
    #    np.savetxt(save_path+'weld_js_exe.csv',exe_out,delimiter=',')
    #    np.savetxt(save_path + "rr_ir_data.csv", np.array(ir_process_output), delimiter=",")
    #    # np.savetxt(save_path + "vel_profile.csv", vel_profile, delimiter=",")
    #    # np.savetxt(save_path + "error.csv", np.array(e_all), delimiter=",")
    #    ir_process_output = []


    ## delay right after welding before jogging
    #time.sleep(1)
    ## jog arm1 out of the way
    #if ONLINE:
    #    q_0 = client.getJointAnglesMH(robot.pulse2deg)
    #    q_0[1] = q_0[1] - np.pi / 8
    #    SS.jog2q(np.hstack((q_0, q2, positioner_js[-1])))
    #flame_3d, torch_path, job_no = flame_tracking_stream(
    #    save_path, robot, robot2, positioner, flir_intrinsic
    #)

    #base_thickness = float(input("Enter base thickness: "))
    #for i in range(flame_3d.shape[0]):
    #    flame_3d[i] = R.T @ flame_3d[i]
    #if flame_3d.shape[0] == 0:
    #    height_offset = 6  # this is arbitrary
    #else:
    #    avg_base_height = np.mean(flame_3d[:, 2])
    #    height_offset = base_thickness - avg_base_height

    #try:
    #    print("Average Base Height:", avg_base_height)
    #    print("Height Offset:", height_offset)
    #except:
    #    height_offset = float(input("Enter height offset: ")) # -8.9564 -9.1457

    height_offset = 5.904
    ######## UPDATE HEIGHT OFFSET IN SEPARATE SCRIPT AND CONNECT TO FLIR #######
    input("Fix height offset, then press enter to continue")

    sub=RRN.SubscribeService('rr+tcp://localhost:12182/?service=FLIR_RR_PROCESS')
    ir_process_result=sub.SubscribeWire("ir_process_result")

    ir_process_result.WireValueChanged += ir_process_cb

    ######## NORMAL LAYERS ########
    num_layer_start = int(3)
    num_layer_end = int(108)

    start_dir = True
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
        lam_relative = calc_lam_cs(curve_sliced_relative)
        print(lam_relative.shape)
        print("------Slice Loaded------")

        # read slicing params
        base_thickness = slicing_meta["baselayer_resolution"]
        num_base = slicing_meta["baselayer_num"]

        # initialize feedrate and velocity
        feedrate=160

        # intialize velocity using speed height model
        model = SpeedHeightModel(a=-0.36997977, b=1.21532975)
        v_nom = model.dh2v(slicing_meta["layer_resolution"])

        # generate a nominal height profile for populating
        height_profile = np.ones(slicing_meta["layer_length"])*slicing_meta["layer_resolution"]

        if layer == 1:
            start_dir=True
            height_err= np.zeros(slicing_meta["layer_length"])
        else:
            start_dir = not np.loadtxt(f"{recorded_dir}layer_{layer-1}/start_dir.csv", delimiter=",")

            ir_error_flag = False
            ### Process IR data prev 
            try:
                flame_3d_prev, _, job_no_prev = flame_tracking_stream(
                        f"{recorded_dir}layer_{layer-1}/",
                        robot,
                        robot2,
                        positioner,
                        flir_intrinsic,
                        height_offset
                        )
                if flame_3d_prev.shape[0] == 0:
                    raise ValueError("No flame detected")
            except ValueError as e:
                print(e)
                flame_3d_prev = None
                ir_error_flag = True
                height_err = np.zeros(len(slicing_meta["layer_length"]))
            else:
                averages_prev = avg_by_line(job_no_prev, flame_3d_prev, np.linspace(0,len(rob1_js)-1,len(rob1_js)))
                heights_prev = averages_prev[:,2]
                if start_dir: heights_prev = np.flip(heights_prev)
                # Find Valid datapoints for height correction
                prev_idx = np.argwhere(np.invert(np.isnan(heights_prev)))

                heights_prev = interpolate_heights(height_profile, heights_prev)
                # height error based on the build height of the previous layer
                height_err = np.ones(len(heights_prev))*slicing_meta["layer_resolution"]*(layer-1)-heights_prev

        if start_dir:
            pass
        else:
            rob1_js = np.flip(rob1_js,axis=0)

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


        ### Calculate dh desired based on the target height, and the error in the previous layer.
        dh_d = height_err+slicing_meta["layer_resolution"]

        # jog to start position
        input("Press Enter to jog to start position")
        if ONLINE:
            SS.jog2q(np.hstack((rob1_js[0], q2, positioner_js[0])))


        ######## FILTER ########
        error=0
        lam_cur=0
        q_cmd_all = []
        job_no = []
        dh_prev_all = []
        v_cmds = []
        lstm_pred
        # current correction Index
        v_cor_idx = 0
        v_corr = 0

        ######## SET INITIAL V #######
        v_cmd=v_nom

        ######## INIT LSTM #######
        lstm = torch.load('../multi_output/saved_model_8_next_step.pt')
        lstm.eval()
        
        # reg parameters
        mean = torch.load('mean.pt')
        std = torch.load('std.pt')

        h = torch.zeros(1,HID_DIM)
        c = torch.zeros(1,HID_DIM)
        state = (h,c)
        u_prev = v_cmd # v_nom
        T_prev = 0.0
        dh_prev = 0.0

        # run one iteration to load the model into cache
        # first iteration takes too long

        _, _ = model(torch.unsqueeze(torch.zeros(3), dim=0),hidden_state = state)

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
            seg_idx = np.where(lam_relative<=lam_cur)[0][-1]
            # checks if we have moved onto the next motion segment
            if seg_idx != v_cor_idx:
                v_cor_idx = seg_idx
                flame_3d = pos_filter.read_filter()
                avg_temp = temp_filter.read_filter()
                if flame_3d[0]!=0:
                    dh_prev = flame_3d[2]-heights_prev[v_cor_idx-1]
                    # print(error)
                    dh_prev_all.append(dh_prev)

                # calculate linearization
                dh_prev = (dh_prev-mean[3])/std[3]
                T_prev = (avg_temp-mean[2])/std[2]
                u_prev = (v_cmd-mean[0])/std[0]
                h_0 = torch.squeeze(state[0])
                c_0 = torch.squeeze(state[1])
                u_0 = torch.tensor([u_prev, T_prev, dh_prev])

                y_0, _ = model(torch.unsqueeze(u_0, dim=0), 
                                    hidden_state=state)
                y_0 = torch.squeeze(y_0)

                A,B,C = lstm_linearization(model, h_0, c_0, u_0)

                # isolate the effect of the velocity on the height input
                B = B[:,0]
                C = C[1,:]

                # generate velocity profile according to optimization
                y_d = torch.unsqueeze(dh_d[v_cor_idx], dim=0)

                u_cmd = (y_d-y_0[1])/(C@B)+u_0[0]
                # convert back from regularization
                v_cmd = u_cmd*std[0]+mean[0]

                # project into valid region
                v_cmd = min(max(v_cmd, V_LOWER) , V_UPPER)

                # propagate the network
                x = torch.unsqueeze(torch.tensor([u_cmd, T_prev, dh_prev]),dim=0)
                y_out, state = model(x, hidden_state=state)

                lstm_pred.append(y_out)

            v_cmds.append(v_cmd)
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
            job_no.append(seg_idx)

            # this function has a delay when loop_start is passed in. 
            # Ensures the update frequency is consistent
            # if (loop_start-time.perf_counter())>1/STREAMING_RATE: 
            #     print("Stopping: Loop Time exceeded streaming period")
            #     break

            # adding delay to counteract delay in streaming send
            if ONLINE: SS.position_cmd(q_cmd, loop_start+DELAY_CORRECTION) 
        if ARCON:
            fronius_client.stop_weld()
        print(f"-----End of Layer {layer}-----")
        if RECORDING:
            js_recording = SS.stop_recording()
            rr_sensors.stop_all_sensors()
            rr_sensors.save_all_sensors(save_path)
            job_no = np.reshape(np.array(job_no), (-1,1))
            q_cmd_all = np.array(q_cmd_all)
            js_recording = np.array(js_recording)
            cmd_out = np.hstack((
                np.reshape(q_cmd_all[:,0],(-1,1)),
                job_no,
                q_cmd_all[:,1:]
                ))
            job_no_exe = []
            # find closest jobno for exe
            for t in js_recording[:,0]:
                idx = np.argmin(np.abs(cmd_out[:,0]-t))
                job_no_exe.append(cmd_out[idx,1])

            job_no_exe = np.reshape(np.array(job_no_exe), (-1,1))
            exe_out = np.hstack((
                np.reshape(js_recording[:,0],(-1,1)),
                job_no_exe,
                js_recording[:,1:]
                ))
            np.savetxt(save_path+'weld_js_cmd.csv',cmd_out,delimiter=',')
            np.savetxt(save_path+'weld_js_exe.csv',exe_out,delimiter=',')
            np.savetxt(save_path + "start_dir.csv", [start_dir], delimiter=",")
            np.savetxt(save_path + "rr_ir_data.csv", np.array(ir_process_output), delimiter=",")
            np.savetxt(save_path + "dh_prev_all.csv", np.array(dh_prev_all), delimiter=",")
            np.savetxt(save_path + "v_cmd.csv", np.array(v_cmds), delimiter=",")
            np.savetxt(save_path + "lstm_pred.csv", np.array(lstm_pred), delimiter=",")

            ir_process_output = []

        # delay right after welding before jogging
        time.sleep(1)
        # jog arm1 out of the way
        if ONLINE:
            q_0 = client.getJointAnglesMH(robot.pulse2deg)
            q_0[1] = q_0[1] - np.pi / 8
            SS.jog2q(np.hstack((q_0, q2, positioner_js[-1])))

        time.sleep(15)


if __name__ == '__main__':
    main()
