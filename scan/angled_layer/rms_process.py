import sys
import yaml
import matplotlib.pyplot as plt
import numpy as np
from motoman_def import robot_obj, positioner_obj
from robotics_utils import H_inv

sys.path.append('../../toolbox')
from angled_layers import *


recorded_dir = '../../../recorded_data/ER4043_bent_tube_2024_08_22_11_12_27/'

config_dir = "../../config/"
flir_intrinsic = yaml.load(open(config_dir + "FLIR_A320.yaml"), Loader=yaml.FullLoader)
dataset = "bent_tube/"
sliced_alg = "slice_ER_4043/"
data_dir = "../../data/" + dataset + sliced_alg

with open(data_dir + "slicing.yml", "r") as file:
    slicing_meta = yaml.safe_load(file)

robot = robot_obj(
    "MA2010_A0",
    def_path="../../config/MA2010_A0_robot_default_config.yml",
    tool_file_path="../../config/torch.csv",
    pulse2deg_file_path="../../config/MA2010_A0_pulse2deg_real.csv",
    d=15,
)
robot2 = robot_obj(
    "MA1440_A0",
    def_path="../../config/MA1440_A0_robot_default_config.yml",
    tool_file_path="../../config/flir.csv",
    pulse2deg_file_path="../../config/MA1440_A0_pulse2deg_real.csv",
    base_transformation_file="../../config/MA1440_pose.csv",
)
positioner = positioner_obj(
    "D500B",
    def_path="../../config/D500B_robot_extended_config.yml",
    tool_file_path="../../config/positioner_tcp.csv",
    pulse2deg_file_path="../../config/D500B_pulse2deg_real.csv",
    base_transformation_file="../../config/D500B_pose.csv",
)

H2010_1440 = H_inv(robot2.base_H)
H = np.loadtxt(data_dir + "curve_pose.csv", delimiter=",")
p = H[:3, -1]
R = H[:3, :3]

height_offset = -4.51 #float(input("Enter height offset: "))
job_no_offset = 3
point_of_rotation = np.array(
        (slicing_meta["point_of_rotation"], slicing_meta["baselayer_thickness"]))
base_thickness = slicing_meta["baselayer_thickness"]
layer_angle = np.array((slicing_meta["layer_angle"]))

num_layer_start = 1
num_layer_end = 10
heights = []
flames = []
for layer in range(num_layer_start, num_layer_end):
    ### Load Data
    curve_sliced_js = np.loadtxt(
        data_dir + f"curve_sliced_js/MA2010_js{layer}_0.csv", delimiter=","
    ).reshape((-1, 6))

    positioner_js = np.loadtxt(
        data_dir + f"curve_sliced_js/D500B_js{layer}_0.csv", delimiter=","
    )
    curve_sliced_relative = np.loadtxt(
        data_dir + f"curve_sliced_relative/slice{layer}_0.csv", delimiter=","
    )
    curve_sliced = np.loadtxt(
        data_dir + f"curve_sliced/slice{layer}_0.csv", delimiter=","
    )
    to_flat_angle = np.deg2rad(layer_angle * (layer - 1))
    dh_max = slicing_meta["dh_max"]
    dh_min = slicing_meta["dh_min"]
    
    ##calculate distance to point of rotation
    dist_to_por = []
    for i in range(len(curve_sliced)):
        point = np.array((curve_sliced[i, 0], curve_sliced[i, 2]))
        dist = np.linalg.norm(point - point_of_rotation)
        dist_to_por.append(dist)

    try:
        flame_3d, _, job_no = flame_tracking(f"{recorded_dir}layer_{layer}/", robot, robot2, positioner, flir_intrinsic, height_offset)
        flames.append(flame_3d)
        if flame_3d.shape[0] == 0:
            raise ValueError("No flame detected")
    except ValueError as e:
        print(e)
        flame_3d= None
    else:
    # rotate to flat
        for i in range(flame_3d.shape[0]):
            flame_3d[i] = R.T @ flame_3d[i] 

            new_x, new_z = rotate(
                point_of_rotation, (flame_3d[:, 0], flame_3d[:, 2]), to_flat_angle
                )
            flame_3d[:, 0] = new_x
            flame_3d[:, 2] = new_z - base_thickness

        job_no= [i - job_no_offset for i in job_no]
        averages= avg_by_line(job_no, flame_3d, np.linspace(0,len(curve_sliced_js)-1,len(curve_sliced_js)))
        heights.append(averages[:,2])
fig = plt.figure()
ax = plt.axes(projection = '3d')
for flame in flames:
    ax.scatter(flame[:,0], flame[:,1], flame[:,2])
plt.show()
for height in heights:
    plt.plot(height)
plt.show()
