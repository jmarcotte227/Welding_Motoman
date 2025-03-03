import pickle
import matplotlib.pyplot as plt
import yaml
import sys
import numpy as np
from motoman_def import robot_obj, positioner_obj
from robotics_utils import H_inv
sys.path.append('../../../toolbox')
from angled_layers import avg_by_line, rotate

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
dataset = "s_curve_angled/"
sliced_alg = "slice/"
data_dir = "../../../data/" + dataset + sliced_alg
mid_layer = 53

# flame_set = 'processing_data/ER4043_bent_tube_2024_09_04_12_23_40_flame.pkl'
flame_set = [
    '../processing_data/s_curve_angled_2025_02_18_11_01_10_flame.pkl'
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
print(p)

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
    point_of_rotation_2 = np.array(
        (slicing_meta["point_of_rotation_2_x"], slicing_meta["point_of_rotation_2_y"])
    )
    base_thickness = slicing_meta["baselayer_thickness"]
    layer_angle = np.array((slicing_meta["layer_angle"]))
    print(layer_angle)
    mid_angle = np.array((slicing_meta["mid_angle"]))

    # curve_sliced = np.loadtxt(data_dir+"curve_sliced/slice1_0.csv", delimiter=',')
    # dist_to_por = []
    # for i in range(len(curve_sliced)):
    #     point = np.array((curve_sliced[i, 0], curve_sliced[i, 2]))
    #     dist = np.linalg.norm(point - point_of_rotation)
    #     dist_to_por.append(dist)

    # height_profile = []
    # for distance in dist_to_por:
    #     height_profile.append(distance * np.sin(np.deg2rad(layer_angle)))
    height_err = []
    height_err_trim = []
    flames_flat = []
    # for layer, flame in enumerate(flames):
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    for layer, flame in enumerate(flames):
        to_flat_angle = np.deg2rad(layer_angle*(layer+1))
        for i in range(flame.shape[0]):
            flame[i,1:] = R.T @ (flame[i,1:]-p)
        # ax.plot(flame[:,1], flame[:,2], flame[:,3])

        if layer>= mid_layer:
                new_x, new_z = rotate(
                    point_of_rotation_2, (flame[:,1], flame[:,3]), -np.deg2rad(layer_angle*(layer+2-mid_layer))
                )
                flame[:, 1] = new_x
                flame[:, 3] = new_z 
                new_x, new_z = rotate(
                    point_of_rotation, (flame[:, 1], flame[:, 3]), np.deg2rad(mid_angle-layer_angle)
                )
                flame[:, 1] = new_x
                flame[:, 3] = new_z - base_thickness

        else:
            new_x, new_z = rotate(
                point_of_rotation, (flame[:, 1], flame[:, 3]), to_flat_angle
            )
            flame[:, 1] = new_x
            flame[:, 3] = new_z - base_thickness

        # print(flame[:,0])
        flame[:,0] = flame[:,0]-job_no_offset
        averages= avg_by_line(flame[:,0], flame[:,1:], np.linspace(0,49,50))
        # print(averages)
        height_err.append(averages[:,2])
        flames_flat.append(averages)
    rms_err = []
    for scan in height_err:
        # print(len(scan[1:-1]))
        rms_err.append(rms_error(scan[1:-1]))
        height_err_trim.append(scan[1:-1])
    rms_errs.append(rms_err)

    print(len(rms_errs[0]))
    plt.plot(rms_errs[0])

    plt.show()
    print("Final Layer Angle: ", rms_errs[0][-1])

np.savetxt(title+'_err.csv',rms_errs[0])
np.savetxt(title+'_layer_err.csv',height_err_trim, delimiter=',')
