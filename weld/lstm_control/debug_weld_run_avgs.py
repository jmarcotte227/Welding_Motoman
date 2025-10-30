import sys, yaml
import numpy as np
import matplotlib.pyplot as plt
from motoman_def import robot_obj, positioner_obj
from tqdm import tqdm
import torch
from qpsolvers import solve_qp

from model_utils import DataReg
from linearization import lstm_linearization

sys.path.append("../../toolbox")
from angled_layers import SpeedHeightModel, flame_tracking_stream,  \
    avg_by_line, interpolate_heights, LiveAverageFilterPos,         \
    LiveAverageFilterScalar
from weld_lstm_ns import WeldLSTMNextStep

NUM_LAYERS = 15

DATASET = 'wall/'
SLICED_ALG = '1_5mm_slice/'
DATA_DIR='../../data/'+DATASET+SLICED_ALG
CONT_MODEL='model_h-8_part-1_loss-0.0411'
V_MIN = torch.tensor(3.0) # mm/s
V_MAX = torch.tensor(17.0) # mm/s
DV_MAX = 3 # mm/s
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

BETA = 0.2
ALPHA = 1.0

REC_DATA = '../../../recorded_data/wall_lstm_control_2025_10_28_16_02_17/'


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

with open(DATA_DIR+'sliced_meta.yml', 'r') as file:
    slicing_meta = yaml.safe_load(file)

# set up figures
fig,ax = plt.subplots(2,1)

height_offset = -8.662751637798227
for layer in tqdm(range(10,NUM_LAYERS)):
    ######## INITIALIZE SAVE DIR #######
    save_path = REC_DATA + f"layer_{layer}/"
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
    else:
        start_dir = not np.loadtxt(f"{REC_DATA}layer_{layer-1}/start_dir.csv", delimiter=",")

        ir_error_flag = False

        ### SIMULATION PROCESS IR DATA NOW ###
        try:
            flame_3d, _, job_no= flame_tracking_stream(
                    f"{REC_DATA}layer_{layer}/",
                    robot,
                    robot2,
                    positioner,
                    flir_intrinsic,
                    height_offset
                    )
            if flame_3d.shape[0] == 0:
                raise ValueError("No flame detected")
        except ValueError as e:
            flame_3d= None
            ir_error= True
            height_err = np.zeros(slicing_meta["layer_length"])
        else:
            averages= avg_by_line(job_no, flame_3d, np.linspace(0,len(rob1_js)-1,len(rob1_js)))
            print(averages)
            heights_prev = averages[:,2]
        # velocities
        v_cmd = np.loadtxt(f"{REC_DATA}layer_{layer}/v_cmd.csv", delimiter=",")

        js_exe = np.loadtxt(f"{REC_DATA}layer_{layer}/weld_js_cmd.csv", delimiter=",")

        job_no = np.linspace(0,48, 49)

        v_cmds = []
        for num in job_no:
            idx = np.where(js_exe[:,1]==num)[0][0]
            v_cmds.append(v_cmd[idx+1])
        ### SIM END ###
        ### Alternative method ###
        dh_prevs = np.loadtxt(f"{REC_DATA}layer_{layer}/dh_prev_all.csv", delimiter=',')
        ### Process IR data prev 
        try:
            flame_3d_prev, _, job_no_prev = flame_tracking_stream(
                    f"{REC_DATA}layer_{layer-1}/",
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
            print(f"H: {height_profile}")
            print(f"H prev pre int: {heights_prev}")
            heights_prev = interpolate_heights(height_profile, heights_prev)
            print(f"H prev: {heights_prev}")
            # height error based on the build height of the previous layer
            height_err = np.ones(len(heights_prev))*build_height-heights_prev
            print(f"H err: {height_err}")

    if start_dir:
        pass
    else:
        rob1_js = np.flip(rob1_js,axis=0)
        rob2_js = np.flip(rob2_js,axis=0)

    ### Calculate dh desired based on the target height, and the error in the previous layer.
    dh_d = torch.tensor(ALPHA*height_err+slicing_meta["layer_resolution"])
    print(f"dH_d: {dh_d}")
    print(f"Layer: {layer}")

    ######## FILTER ########
    error=0
    lam_cur=0
    q_cmd_all = []
    job_no = []
    dh_prev_all = []
    v_cmds = []
    lstm_pred = []
    # current correction Index
    v_cor_idx = 0
    v_corr = 0

    ######## SET INITIAL V #######
    v_cmd = v_nom
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
    y_d[0,:] = reg.reg(dh_d[v_cor_idx]-1,1)

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
    v_cmd = float(u_cmd)

    v_cmds.append(v_cmd)

    # propagate the network
    x = reg.reg(torch.cat((u_cmd, dh_prev), dim=0))
    y_out, state = lstm(torch.unsqueeze(x, dim=0), hidden_state=state)


    lstm_pred.append(torch.squeeze(y_out.detach()))

    print(v_cmd)

    print(v_max)
    # v_max = v_max.detach().numpy().astype("double")
    print(reg.unreg(v_max,0))
    exit()

    # Looping through the entire path of the sliced part
    for seg_idx in range(len(rob1_js)-2):
        flame_3d = averages[seg_idx,:]
        # dh_prev = torch.unsqueeze(torch.tensor(flame_3d[2]-heights_prev[seg_idx], dtype=torch.float32), dim=0)
        dh_prev = torch.unsqueeze(torch.tensor(dh_prevs[seg_idx], dtype=torch.float32), dim=0)
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
        u_cmd = torch.tensor()

        # propagate the network
        x = reg.reg(torch.cat((u_cmd, dh_prev), dim=0))
        y_out, state = lstm(torch.unsqueeze(x, dim=0), hidden_state=state)


        v_cmd = float(u_cmd)


        lstm_pred.append(torch.squeeze(y_out.detach()))

        v_cmds.append(v_cmd)

    print(v_cmds)
    # print(np.array(lstm_pred))
    exit()





    # ax[0].plot(averages[:,-1], 'b')
    # ax[0].plot(height_profile, 'r')
    # ax[1].plot(vels)
plt.show()
