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

from qpsolvers import solve_qp
sys.path.append("../../toolbox")
from angled_layers import SpeedHeightModel, flame_tracking_stream,  \
    avg_by_line, interpolate_heights, LiveAverageFilterPos,         \
    LiveAverageFilterScalar
from lstm_model_next_step_fast import WeldLSTM
from linearization import lstm_linearization
from model_utils import DataReg
from weld_lstm_ns import WeldLSTMNextStep

ir_updated_flag = False
ir_process_packet = None
ir_process_output = []

# Initialize Filters
pos_filter = LiveAverageFilterPos()

def ir_process_cb(sub, value, ts):
    global ir_updated_flag, ir_process_packet, ir_process_output, pos_filter
    ir_process_packet=copy.deepcopy(value)
    ir_process_output.append(ir_process_packet.flame_position)
    # add z position to averaging filter
    pos_filter.log_reading(ir_process_packet.flame_position)
    ir_updated_flag=True

def main():

    ######## IR VARIABLES #########
    global ir_updated_flag, ir_process_packet, ir_process_output, pos_filter

    ######## Welding Parameters ########
    ARCON = True
    BASE_LAYERS = False
    RECORDING = True
    ONLINE = True # Used to test without connecting to RR services
    BASE_VEL = 3
    BASE_FEEDRATE= 300
    FEEDRATE = 160
    JOB_OFFSET = 200
    STREAMING_RATE = 125.

    DELAY_CORRECTION = 0.0007

    DATASET = 'wall/'
    SLICED_ALG = '1_5mm_slice/'
    DATA_DIR='../../data/'+DATASET+SLICED_ALG
    CONT_MODEL='model_h-8_part-1_loss-0.0411'

    with open(DATA_DIR+'sliced_meta.yml', 'r') as file:
        slicing_meta = yaml.safe_load(file)

    ####### CONTROLLER PARAMETERS #######
    # V_GAIN = 3.612# changed from this at layer 72 700.39922 # gradient of model at 3mm/s lower bound
    V_MIN = torch.tensor(3.0) # mm/s
    V_MAX = torch.tensor(17.0) # mm/s
    DV_MAX = 3 # mm/s

    BETA = 0.2
    ALPHA = 1.0

    ######## Create Directories ########
    # now = datetime.now()
    # recorded_dir = now.strftime(
    #     "../../../recorded_data/wall_lstm_control_%Y_%m_%d_%H_%M_%S/"
    # )
    # os.makedirs(recorded_dir)
    recorded_dir = "../../../recorded_data/wall_lstm_control_2025_10_31_13_34_50/"

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
    if BASE_LAYERS:
        for layer in range(0,slicing_meta["baselayer_num"]):
            save_path = recorded_dir + f"baselayer_{layer}/"
            os.mkdir(save_path)

            ######### LOAD POINT DATA ########
            rob1_js = np.loadtxt(
                DATA_DIR+f'curve_sliced_js/MA2010_base_js{layer}_0.csv',
                delimiter=','
            )
            rob2_js = np.loadtxt(
                DATA_DIR+f'curve_sliced_js/MA1440_base_js{layer}_0.csv',
                delimiter=','
            )
            positioner_js = np.loadtxt(
                DATA_DIR+f'curve_sliced_js/D500B_base_js{layer}_0.csv',
                delimiter=','
            )
            curve_sliced_relative = np.loadtxt(
                DATA_DIR+f'curve_sliced_relative/baselayer{layer}_0.csv',
                delimiter=','
            )
            lam_relative = calc_lam_cs(curve_sliced_relative)
            print("------Slice Loaded------")
            if layer==0:
                pass
            else:
                rob1_js = np.flip(rob1_js,axis=0)
                rob2_js = np.flip(rob2_js,axis=0)

            ## jog to start position
            input("Press Enter to jog to start position")
            if ONLINE:
                SS.jog2q(np.hstack((rob1_js[0], rob2_js[0], positioner_js[0])))

            lam_cur=0
            q_cmd_all = []
            job_no = []
            v_cmd = BASE_VEL

            input("press enter to start layer")

            if RECORDING:
                rr_sensors.start_all_sensors()
                SS.start_recording()
            if ARCON:
                fronius_client.job_number = int(BASE_FEEDRATE/10+JOB_OFFSET)
                fronius_client.start_weld()
            while lam_cur<lam_relative[-1] - v_cmd/STREAMING_RATE:
                loop_start = time.perf_counter()

                # calculate nominal vel of segment
                seg_idx = np.where(lam_relative<=lam_cur)[0][-1]

                lam_cur += v_cmd/STREAMING_RATE
                # get closest lambda that is greater than current lambda
                lam_idx = np.where(lam_relative>=lam_cur)[0][0]
                # Calculate the fraction of the current lambda that has been traversed
                lam_ratio = ((lam_cur-lam_relative[lam_idx-1])/
                             (lam_relative[lam_idx]-lam_relative[lam_idx-1]))
                # Apply that fraction to the joint space
                q1 = rob1_js[lam_idx-1]*(1-lam_ratio)+rob1_js[lam_idx]*lam_ratio
                q2 = rob2_js[lam_idx-1]*(1-lam_ratio)+rob2_js[lam_idx]*lam_ratio
                q_positioner = positioner_js[lam_idx-1]*(1-lam_ratio) + positioner_js[lam_idx]*(lam_ratio)

                #generate set of joints to command
                q_cmd = np.hstack((q1, q2, q_positioner))

                # log q_cmd
                q_cmd_all.append(np.hstack((time.perf_counter(),q_cmd)))
                job_no.append(seg_idx)

                # this function has a delay when loop_start is passed in.
                # Ensures the update frequency is consistent
                if (loop_start-time.perf_counter())>1/STREAMING_RATE:
                    print("Stopping: Loop Time exceeded streaming period")
                    break

                # adding delay to counteract delay in streaming send
                if ONLINE: 
                    SS.position_cmd(q_cmd, loop_start+DELAY_CORRECTION) 
            if ARCON:
                fronius_client.stop_weld()
            print(f"-----End of Base Layer {layer}-----")
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
                np.savetxt(save_path + "rr_ir_data.csv", np.array(ir_process_output), delimiter=",")
                ir_process_output = []


            # delay right after welding before jogging
            time.sleep(1)
            # jog arm1 out of the way
            if ONLINE:
                q_0 = client.getJointAnglesMH(robot.pulse2deg)
                q_0[1] = q_0[1] - np.pi / 8
                SS.jog2q(np.hstack((q_0, q2, positioner_js[-1])))

        flame_3d, torch_path, job_no = flame_tracking_stream(
            save_path, robot, robot2, positioner, flir_intrinsic
        )

        ### CALCULATING HEIGHT OFFSET OF TOTAL BASE LAYER HEIGHT ###
        flame_3d, torch_path, job_no = flame_tracking_stream(
            save_path, robot, robot2, positioner, flir_intrinsic
        )

        base_thickness = float(input("Enter base thickness: "))
        if flame_3d.shape[0] == 0:
            height_offset = 3.6  # this is arbitrary
        else:
            avg_base_height = np.mean(flame_3d[:, 2])
            height_offset = base_thickness - avg_base_height

    try:
        print("Average Base Height:", avg_base_height)
        print("Height Offset:", height_offset)
    except:
        # height_offset = float(input("Enter height offset: ")) 
        height_offset = -7.92870911432761
        print("height offset set manually")

    ######## UPDATE HEIGHT OFFSET IN SEPARATE SCRIPT AND CONNECT TO FLIR #######
    input("Fix height offset, then press enter to continue")

    sub=RRN.SubscribeService('rr+tcp://localhost:12182/?service=FLIR_RR_PROCESS')
    ir_process_result=sub.SubscribeWire("ir_process_result")

    ir_process_result.WireValueChanged += ir_process_cb

    ######## NORMAL LAYERS ########
    num_layer_start = int(0)
    num_layer_end = int(50)

    start_dir = True
    for layer in range(num_layer_start, num_layer_end):
        ######## INITIALIZE SAVE DIR #######
        save_path = recorded_dir + f"layer_{layer}/"
        os.mkdir(save_path)
        ######## LOAD POINT DATA ########
        rob1_js = np.loadtxt(
            DATA_DIR+f'curve_sliced_js/MA2010_js{layer}_0.csv',
            delimiter=','
        )
        rob2_js = np.loadtxt(
            DATA_DIR+f'curve_sliced_js/MA1440_js{layer}_0.csv',
            delimiter=','
        )
        positioner_js = np.loadtxt(
            DATA_DIR+f'curve_sliced_js/D500B_js{layer}_0.csv', 
            delimiter=','
        )
        curve_sliced_relative = np.loadtxt(
            DATA_DIR+f'curve_sliced_relative/slice{layer}_0.csv',
            delimiter=','
        )
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

        build_height = layer*slicing_meta["layer_resolution"]\
                        +slicing_meta["baselayer_num"]*slicing_meta["baselayer_resolution"]
        height_profile = np.ones(slicing_meta["layer_length"])*build_height

        if layer == 0:
            start_dir=True
            height_err= np.zeros(slicing_meta["layer_length"])
            try:
                flame_3d_prev, _, job_no_prev = flame_tracking_stream(
                        f"{recorded_dir}baselayer_1/",
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
                height_err = np.zeros(slicing_meta["layer_length"])
            else:
                averages_prev = avg_by_line(job_no_prev, flame_3d_prev, np.linspace(0,len(rob1_js)-1,len(rob1_js)))
                heights_prev = averages_prev[:,2]
                if start_dir: heights_prev = np.flip(heights_prev)

                # TODO fix this error
                print(height_profile)
                print(heights_prev)
                heights_prev = interpolate_heights(height_profile, heights_prev)
                print(heights_prev)
                # height error based on the build height of the previous layer
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
                height_err = np.zeros(slicing_meta["layer_length"])
            else:
                averages_prev = avg_by_line(job_no_prev, flame_3d_prev, np.linspace(0,len(rob1_js)-1,len(rob1_js)))
                heights_prev = averages_prev[:,2]
                if start_dir: heights_prev = np.flip(heights_prev)

                # TODO fix this error
                print(height_profile)
                print(heights_prev)
                heights_prev = interpolate_heights(height_profile, heights_prev)
                print(heights_prev)
                # height error based on the build height of the previous layer
                height_err = np.ones(len(heights_prev))*build_height-heights_prev
                print(height_err)

        if start_dir:
            pass
        else:
            rob1_js = np.flip(rob1_js,axis=0)
            rob2_js = np.flip(rob2_js,axis=0)

        ### Calculate dh desired based on the target height, and the error in the previous layer.
        dh_d = torch.tensor(ALPHA*height_err+slicing_meta["layer_resolution"])
        print(dh_d)
        np.savetxt(save_path+'dh_d.csv',dh_d.detach().numpy(),delimiter=',')

        # jog to start position
        input("Press Enter to jog to start position")
        if ONLINE:
            SS.jog2q(np.hstack((rob1_js[0], rob2_js[0], positioner_js[0])))

        ######## FILTER ########
        error=0
        lam_cur=0
        q_cmd_all = []
        job_no = []
        dh_prev_all = []
        T_prev_all = []
        v_cmds = []
        v_cor_idxs = []
        lstm_pred = []
        filt_ir_height = []
        # current correction Index
        v_cor_idx = 0

        ######## SET INITIAL V #######
        v_cmd=v_nom
        u_cmd = torch.tensor(v_cmd, dtype=torch.float32)

        ######## INIT LSTM #######
        device = torch.device("cpu")
        cont_data = torch.load(f'{CONT_MODEL}/{CONT_MODEL}.pt')
        lstm = WeldLSTMNextStep(
            cont_data["input_dim"],
            cont_data["hidden_dim"],
            cont_data["output_dim"],
            cont_data["num_layers"],
            cont_data["dropout"],
        )
        lstm.load_state_dict(cont_data["model_state_dict"])
        lstm.to(device)
        lstm.eval()


        # initialize regularizer
        reg = DataReg(cont_data["data_mean"], cont_data["data_std"])

        # convert the limits
        v_min = torch.tensor([reg.reg(V_MIN,0)], dtype=torch.float32)
        v_max = torch.tensor([reg.reg(V_MAX,0)], dtype=torch.float32)
        dv_max = torch.tensor([reg.scale(DV_MAX,0)], dtype=torch.float32)

        
        # initialize the hidden state
        h = torch.zeros(1,cont_data["hidden_dim"])
        c = torch.zeros(1,cont_data["hidden_dim"])
        state = (h,c)

        # initialize previous inputs to the mean
        u_prev = cont_data["data_mean"][0].reshape(1)
        dh_prev = cont_data["data_mean"][1].reshape(1)

        # setup QP params
        Q = torch.tensor([[1.0]])
        Q_delta = torch.tensor([[BETA]])

        # generate first velocity command
        # TODO: run the linearization once
        h_0_cont = torch.squeeze(state[0])
        c_0_cont = torch.squeeze(state[1])
        x_0_cont = reg.reg(torch.cat((u_prev, dh_prev), dim=0))

        y_0_cont, _ = lstm(torch.unsqueeze(x_0_cont, dim=0), 
                            hidden_state=state)
        y_0_cont = torch.unsqueeze(y_0_cont, dim=0)

        _,B,C = lstm_linearization(lstm, h_0_cont, c_0_cont, x_0_cont)

        # only look at the column changed by the control input
        B = torch.unsqueeze(B[:,0], dim=-1)

        #### QP Control ####
        y_d = torch.zeros((1,1))
        y_d[0,:] = reg.reg(dh_d[v_cor_idx],1)

        P= (B.T@C.T@Q@C@B+0.5*Q_delta).detach().numpy().astype("double")
        q = ((y_0_cont-(C@B*x_0_cont[0])-y_d).T@Q@C@B-x_0_cont[0]*Q_delta).detach().numpy().astype("double")
        G = np.array([[1.0],
                      [-1.0]]).astype("double")
        h = np.array([[dv_max[0]+x_0_cont[0].detach()],
                      [dv_max[0]-x_0_cont[0].detach()]]).astype("double")
        lb = v_min.detach().numpy().astype("double")
        ub = v_max.detach().numpy().astype("double")

        u_cmd_cont = solve_qp(P,q,G,h,lb=lb,ub=ub, solver='quadprog', verbose=True)

        u_cmd_cont = torch.tensor(u_cmd_cont, dtype=torch.float32)
        u_cmd = reg.unreg(u_cmd_cont, 0)

        # propagate the network
        x = reg.reg(torch.cat((u_cmd, dh_prev), dim=0))
        y_out, state = lstm(torch.unsqueeze(x, dim=0), hidden_state=state)

        v_cmd = float(u_cmd)


        lstm_pred.append(torch.squeeze(y_out.detach()))

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

            # calculate which index we are on
            seg_idx = np.where(lam_relative<=lam_cur)[0][-1]
            # checks if we have moved onto the next motion segment
            if seg_idx != v_cor_idx:
                v_cor_idx = seg_idx
                flame_3d = pos_filter.read_filter()
                filt_ir_height.append(np.squeeze(flame_3d))
                # print(flame_3d)
                if flame_3d[0]!=0:
                    dh_prev = torch.tensor(flame_3d[2]-heights_prev[v_cor_idx-1], dtype=torch.float32)
                    # print(error)
                else:
                    print("flame error")
                    print(flame_3d)
                    dh_prev = cont_data["data_mean"][1].reshape(1)
                dh_prev_all.append(dh_prev)
                # calculate linearization
                h_0_cont = torch.squeeze(state[0])
                c_0_cont = torch.squeeze(state[1])
                x_0_cont = reg.reg(torch.cat((u_prev, dh_prev), dim=0))

                y_0_cont, _ = lstm(torch.unsqueeze(x_0_cont, dim=0), 
                                    hidden_state=state)
                y_0_cont = torch.unsqueeze(y_0_cont, dim=0)

                _,B,C = lstm_linearization(lstm, h_0_cont, c_0_cont, x_0_cont)

                # only look at the column changed by the control input
                B = torch.unsqueeze(B[:,0], dim=-1)

                #### QP Control ####
                y_d = torch.zeros((1,1))
                y_d[0,:] = reg.reg(dh_d[v_cor_idx],1)
                # print("y")
                # print(y_d)

                P= (B.T@C.T@Q@C@B+0.5*Q_delta).detach().numpy().astype("double")
                q = ((y_0_cont-(C@B*x_0_cont[0])-y_d).T@Q@C@B-x_0_cont[0]*Q_delta).detach().numpy().astype("double")
                G = np.array([[1.0],
                              [-1.0]]).astype("double")
                h = np.array([[dv_max[0]+x_0_cont[0].detach()],
                              [dv_max[0]-x_0_cont[0].detach()]]).astype("double")
                lb = v_min.detach().numpy().astype("double")
                ub = v_max.detach().numpy().astype("double")

                u_cmd_cont = solve_qp(P,q,G,h,lb=lb,ub=ub, solver='quadprog', verbose=True)

                u_cmd_cont = torch.tensor(u_cmd_cont, dtype=torch.float32)
                u_cmd = reg.unreg(u_cmd_cont, 0)
                u_prev = u_cmd

                # propagate the network
                x_cont = reg.reg(torch.cat((u_cmd, dh_prev), dim=0))
                y_out_cont, state = lstm(torch.unsqueeze(x_cont, dim=0), hidden_state=state)

                v_cmd = float(u_cmd)

                lstm_pred.append(torch.squeeze(reg.unreg(y_out_cont.detach(),1)))

            v_cmds.append(v_cmd)
            v_cor_idxs.append(v_cor_idx)
            lam_cur += v_cmd/STREAMING_RATE
            # get closest lambda that is greater than current lambda
            lam_idx = np.where(lam_relative>=lam_cur)[0][0]
            # Calculate the fraction of the current lambda that has been traversed
            lam_ratio = ((lam_cur-lam_relative[lam_idx-1])/
                         (lam_relative[lam_idx]-lam_relative[lam_idx-1]))
            # Apply that fraction to the joint space
            q1 = rob1_js[lam_idx-1]*(1-lam_ratio)+rob1_js[lam_idx]*lam_ratio
            q2 = rob2_js[lam_idx-1]*(1-lam_ratio)+rob2_js[lam_idx]*lam_ratio
            q_positioner = positioner_js[lam_idx-1]*(1-lam_ratio) + positioner_js[lam_idx]*(lam_ratio)

            #generate set of joints to command
            q_cmd = np.hstack((q1, q2, q_positioner))

            # log q_cmd
            q_cmd_all.append(np.hstack((time.perf_counter(),q_cmd)))
            job_no.append(seg_idx)

            # this function has a delay when loop_start is passed in. 
            # Ensures the update frequency is consistent
            if (loop_start-time.perf_counter())>1/STREAMING_RATE: 
                print("Stopping: Loop Time exceeded streaming period")
                break

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
            print(f"V Command: {v_cmds}")
            np.savetxt(save_path+'weld_js_cmd.csv',cmd_out,delimiter=',')
            np.savetxt(save_path+'weld_js_exe.csv',exe_out,delimiter=',')
            np.savetxt(save_path + "start_dir.csv", [start_dir], delimiter=",")
            np.savetxt(save_path + "rr_ir_data.csv", np.array(ir_process_output), delimiter=",")
            np.savetxt(save_path + "dh_prev_all.csv", np.array(dh_prev_all), delimiter=",")
            np.savetxt(save_path + "T_prev_all.csv", np.array(T_prev_all), delimiter=",")
            np.savetxt(save_path + "lstm_pred.csv", np.array(lstm_pred), delimiter=",")
            np.savetxt(save_path + "v_cmd.csv", np.array(v_cmds), delimiter=",")
            np.savetxt(save_path + "v_cor_idx.csv", np.array(v_cor_idxs), delimiter=",")
            np.savetxt(save_path + "filt_ir_height.csv", np.array(filt_ir_height), delimiter=",")

            ir_process_output = []

        # delay right after welding before jogging
        time.sleep(1)

        # jog arm1 out of the way
        if ONLINE:
            q_0 = client.getJointAnglesMH(robot.pulse2deg)
            q_0[1] = q_0[1] - np.pi / 8
            SS.jog2q(np.hstack((q_0, q2, positioner_js[-1])))

        input("enter to continue")
        # time.sleep(15)


if __name__ == '__main__':
    main()
