import time, os, copy, yaml, sys
import numpy as np
from motoman_def import robot_obj, positioner_obj
from lambda_calc import calc_lam_cs
from RobotRaconteur.Client import *
from weldRRSensor import *
from dual_robot import *
from traj_manipulation import *
# from StreamingSend import StreamingSend
from robotics_utils import H_inv, VectorPlaneProjection
from scipy.interpolate import interp1d
from scipy.optimize import minimize, Bounds
from numpy.linalg import norm
from matplotlib import pyplot as plt

sys.path.append("../../toolbox/")
sys.path.append("")
from angled_layers import *
import numpy as np

if __name__ == '__main__':
    LAYER = 45
    HEIGHT_OFFSET = -8.075
    JOB_NO_OFFSET = 3
# import part data
    CONFIG_DIR = '../../config/'
    DATASET = 'bent_tube/'
    SLICED_ALG = 'slice_ER_4043_large_hot/'
    DATA_DIR='../../data/'+DATASET+SLICED_ALG
    with open(DATA_DIR+'slicing.yml', 'r') as file:
        slicing_meta = yaml.safe_load(file)
    # camera parameters
    flir_intrinsic = yaml.load(open(CONFIG_DIR + "FLIR_A320.yaml"), Loader=yaml.FullLoader)
    # robot objects
    robot = robot_obj(
        "MA2010_A0",
        def_path=CONFIG_DIR+"MA2010_A0_robot_default_config.yml",
        tool_file_path=CONFIG_DIR+"torch.csv",
        pulse2deg_file_path=CONFIG_DIR+"MA2010_A0_pulse2deg_real.csv",
        d=15,
    )
    robot2 = robot_obj(
        "MA1440_A0",
        def_path=CONFIG_DIR+"MA1440_A0_robot_default_config.yml",
        tool_file_path=CONFIG_DIR+"flir_imaging.csv",
        pulse2deg_file_path=CONFIG_DIR+"MA1440_A0_pulse2deg_real.csv",
        base_transformation_file=CONFIG_DIR+"MA1440_pose.csv",
    )
    positioner = positioner_obj(
        "D500B",
        def_path=CONFIG_DIR+"D500B_robot_default_config.yml",
        tool_file_path=CONFIG_DIR+"positioner_tcp.csv",
        pulse2deg_file_path=CONFIG_DIR+"D500B_pulse2deg_real.csv",
        base_transformation_file=CONFIG_DIR+"D500B_pose_large_tube.csv",
    )
    base_thickness = slicing_meta["baselayer_thickness"]
    layer_angle = np.array((slicing_meta["layer_angle"]))


    # set up transform
    H2010_1440 = H_inv(robot2.base_H)

    H = np.loadtxt(DATA_DIR + "curve_pose.csv", delimiter=",")
    p = H[:3, -1]
    R = H[:3, :3]

    # import joint data
    curve_sliced_js = np.loadtxt(
        DATA_DIR + f"curve_sliced_js/MA2010_js{LAYER}_0.csv", delimiter=","
    ).reshape((-1, 6))

    positioner_js = np.loadtxt(
        DATA_DIR + f"curve_sliced_js/D500B_js{LAYER}_0.csv", delimiter=","
    )
    curve_sliced_relative = np.loadtxt(
        DATA_DIR + f"curve_sliced_relative/slice{LAYER}_0.csv", delimiter=","
    )
    curve_sliced = np.loadtxt(
        DATA_DIR + f"curve_sliced/slice{LAYER}_0.csv", delimiter=","
    )
    to_flat_angle = np.deg2rad(layer_angle * (LAYER-1))
    dh_max = slicing_meta["dh_max"]
    dh_min = slicing_meta["dh_min"]
    point_of_rotation = np.array(
        (slicing_meta["point_of_rotation"], slicing_meta["baselayer_thickness"])
    )
    model = SpeedHeightModel(a=-0.36997977, b=1.21532975)
    h = [model.v2dh(5+.1), model.v2dh(5-.1)]
    print("Linearizing: ", (h[1]-h[0])/.2)
    # import flame data
    FLAME_DATA_DIR = '../../../recorded_data/ER4043_bent_tube_large_hot_2024_11_06_12_27_19/'
    fig, [ax1,ax2] = plt.subplots(2,1)
    try:
        print("Loading Flame Data")
        flame_3d_prev, _, job_no = flame_tracking(f"{FLAME_DATA_DIR}layer_{LAYER-1}/",
                                                       robot,
                                                       robot2,
                                                       positioner,
                                                       flir_intrinsic,
                                                       HEIGHT_OFFSET)
        print(flame_3d_prev)
        vel_planned = np.loadtxt(FLAME_DATA_DIR+f'layer_{LAYER-1}/velocity_profile.csv', delimiter=',')
        if flame_3d_prev.shape[0] == 0:
            raise ValueError("No flame detected")
    except ValueError as e:
        print(e)
        flame_3d_prev = None
        ir_error_flag = True
    else:
        # rotate to flat
        times = []
        filter = LiveFilter()
        for i in range(flame_3d_prev.shape[0]):
            #filtering flame instead
            filt_out = filter.process(R.T @ flame_3d_prev[i,:])
            print(filt_out)
            flame_3d_prev[i,:] = filt_out
            new_x, new_z = rotate(
                point_of_rotation, (flame_3d_prev[i, 0], flame_3d_prev[i, 2]), to_flat_angle
            )
            flame_3d_prev[i, 0] = new_x
            flame_3d_prev[i, 2] = new_z - base_thickness  

    # vel_planned = np.flip(vel_planned)
    # prune non-job number layers
    del_idx = []
    vel_list = []
    vel_correction = []
    for i, val in enumerate(job_no):
        if val<=JOB_NO_OFFSET+1:
            del_idx.append(i)
        elif val>=50+JOB_NO_OFFSET:
            del_idx.append(i)
        else:
            vel_list.append(vel_planned[val-JOB_NO_OFFSET])
    
    flame_3d_prev=np.delete(flame_3d_prev, del_idx, axis=0)
    error = flame_3d_prev[:,2]

    # mimic filter 
    out = []
    # empty first element to account for sample delay
    vel_correction = [np.nan]
    for e in error:
        # e_filt = filter.process([e])
        # out.append(e_filt)
        out.append(e)
        # vel_correction.append(vel_adjust(e_filt, k=-2)[0])
        vel_correction.append(vel_adjust(e, k=-2))
    # print(vel_correction)

    # calculate correction
    # for i in range(flame_3d_prev.shape[0]):
    #     vel_correction.append(vel_adjust(flame_3d_prev[i,2], k=-2))
    # appending nan to account for sample delay
    vel_list.append(np.nan)
    vel_list = np.array(vel_list)
    vel_correction = np.array(vel_correction)
    # calculate deposition for nominal model
    h_nom = model.v2dh(vel_list)
    h_corr = model.v2dh(vel_list+vel_correction)
    diff_h = h_corr-h_nom

    ax1.plot(flame_3d_prev[:,2])
    ax1.plot(diff_h)
    ax1.legend(["Error", "Added Height"])
    ax2.plot(vel_list)
    ax2.plot(vel_list+vel_correction)
    ax2.legend(["V_nom", "V_corrected"])
    ax1.set_ylim([-2,2])
    plt.show()

# transform and extract error
# simulate part welding, recieving frame every .03 seconds

# compute velocity edit
# tune gain here

# print velocity edit
# plot entire velocity edit vs layer error



