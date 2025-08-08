import pickle
import matplotlib.pyplot as plt
import yaml
import sys
import numpy as np
from motoman_def import robot_obj, positioner_obj
from robotics_utils import H_inv
sys.path.append('../../../toolbox')
from angled_layers import avg_by_line, rotate
from copy import deepcopy

def rms_error(data):
    data = np.array(data)
    n = 0
    num = 0
    for i in data:
        if not np.isnan(i): 
            num = num + i**2
            n+=1
    return np.sqrt(num/n)

config_dir = "../../../config/"
dataset = "bent_tube/"
sliced_alg = "slice_ER_4043_large_hot/"
data_dir = "../../../data/" + dataset + sliced_alg

# flame_set = 'processing_data/ER4043_bent_tube_2024_09_04_12_23_40_flame.pkl'
flame_set = [
    #'../processing_data/ER4043_bent_tube_2024_08_28_12_24_30_flame.pkl',
    # '../processing_data/ER4043_bent_tube_2024_09_04_12_23_40_flame.pkl',
    # 'processing_data/ER4043_bent_tube_2024_09_03_13_26_16_flame.pkl',
    # '../processing_data/ER4043_bent_tube_hot_2024_10_21_13_25_58_flame.pkl'
    '../processing_data/ER4043_bent_tube_large_hot_2024_11_06_12_27_19_flame.pkl'
    # '../processing_data/ER4043_bent_tube_large_cold_2024_11_07_10_21_39_flame.pkl'
    # '../processing_data/ER4043_bent_tube_large_cold_OL_2024_11_14_11_56_43_flame.pkl'
    # '../processing_data/ER4043_bent_tube_large_hot_OL_2024_11_14_13_05_38_flame.pkl'
    # '../processing_data/ER4043_bent_tube_large_hot_streaming_2025_03_06_feedback_troubleshooting_flame.pkl'
]
title=flame_set[-1].removesuffix('_flame.pkl').removeprefix('../processing_data/')
with open(data_dir + "slicing.yml", "r") as file:
    slicing_meta = yaml.safe_load(file)

robot = robot_obj(
    "MA2010_A0",
    def_path=config_dir+"MA2010_A0_robot_default_config.yml",
    tool_file_path=config_dir+"torch.csv",
    pulse2deg_file_path=config_dir+"MA2010_A0_pulse2deg_real.csv",
    d=15,
)
robot2 = robot_obj(
    "MA1440_A0",
    def_path=config_dir+"MA1440_A0_robot_default_config.yml",
    tool_file_path=config_dir+"flir_imaging.csv",
    pulse2deg_file_path=config_dir+"MA1440_A0_pulse2deg_real.csv",
    base_transformation_file=config_dir+"MA1440_pose.csv",
)
positioner = positioner_obj(
    "D500B",
    def_path=config_dir+"D500B_robot_default_config.yml",
    tool_file_path=config_dir+"positioner_tcp.csv",
    pulse2deg_file_path=config_dir+"D500B_pulse2deg_real.csv",
    base_transformation_file=config_dir+"D500B_pose.csv",
)

H2010_1440 = H_inv(robot2.base_H)
H = np.loadtxt(data_dir + "curve_pose.csv", delimiter=",")
p = H[:3, -1]
R = H[:3, :3]

layer_start = 1
rms_errs = []

for idx,flame in enumerate(flame_set):
    with open(flame, 'rb') as file:
        flames = pickle.load(file)
    print("Flames Loaded, plotting")

    # Rotation parameters
    job_no_offset = 3
    point_of_rotation = np.array(
            (slicing_meta["point_of_rotation"], slicing_meta["baselayer_thickness"]))
    base_thickness = slicing_meta["baselayer_thickness"]
    layer_angle = np.array((slicing_meta["layer_angle"]))

    curve_sliced = np.loadtxt(data_dir+"curve_sliced/slice1_0.csv", delimiter=',')
    dist_to_por = []
    for i in range(len(curve_sliced)):
        point = np.array((curve_sliced[i, 0], curve_sliced[i, 2]))
        dist = np.linalg.norm(point - point_of_rotation)
        dist_to_por.append(dist)

    dhs = []
    # append empty list to dhs to 
    nan_list = np.empty(50)
    nan_list[:]=np.nan
    dhs.append(nan_list)
    fig,ax = plt.subplots()
    for layer, flame in enumerate(flames[:-1]):
        # looking ahead to next layer to calculate dh
        flame_next = deepcopy(flames[layer+1])
        to_flat_angle = np.deg2rad(layer_angle*(layer+layer_start))
        for i in range(flame.shape[0]):
            flame[i,1:] = R.T @ flame[i,1:]
        # now the next layer
        for i in range(flame_next.shape[0]):
            flame_next[i,1:] = R.T @ flame_next[i,1:]
        

        # flat flame
        new_x, new_z = rotate(
            point_of_rotation, (flame[:, 1], flame[:, 3]), to_flat_angle
        )
        flame[:, 1] = new_x
        flame[:, 3] = new_z - base_thickness

        flame[:,0] = flame[:,0]-job_no_offset
        # next flame
        new_x, new_z = rotate(
            point_of_rotation, (flame_next[:, 1], flame_next[:, 3]), to_flat_angle
        )
        flame_next[:, 1] = new_x
        flame_next[:, 3] = new_z - base_thickness

        flame_next[:,0] = flame_next[:,0]-job_no_offset
        flat_averages= avg_by_line(flame[:,0], flame[:,1:], np.linspace(0,49,50))
        next_averages= avg_by_line(flame_next[:,0], flame_next[:,1:], np.linspace(0,49,50))

        # flip flame so datapoint matches flame_next position
        flat_averages=np.flip(flat_averages, axis=0)

        dhs.append(next_averages[:,2]-flat_averages[:,2])
        ax.plot(flame[:,1], flame[:,3])
    plt.show()
    print(np.array(dhs))
    np.savetxt(title+'_dhs.csv',dhs, delimiter=',')
