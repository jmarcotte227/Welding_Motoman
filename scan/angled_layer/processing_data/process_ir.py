''' 
Imports IR data, and reads bakc flame position data in the format 
[[[job_no, flame_x, flame_y, flame_z],
  [job_no, flame_x, flame_y, flame_z],
  ...
  [job_no, flame_x, flame_y, flame_z]],
 [[new layer data ...

'''
import sys
import yaml
import matplotlib.pyplot as plt
import numpy as np
from motoman_def import robot_obj, positioner_obj
from robotics_utils import H_inv
import scipy
import matplotlib.animation as animation
import pickle

sys.path.append('../../../toolbox')
from angled_layers import rotate, flame_tracking, avg_by_line, calc_velocity, SpeedHeightModel

config_dir = "../../../config/"
flir_intrinsic = yaml.load(open(config_dir + "FLIR_A320.yaml"), Loader=yaml.FullLoader)
dataset = "bent_tube/"
sliced_alg = "slice_ER_4043_hot/"
data_dir = "../../../data/" + dataset + sliced_alg

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

height_offset = -4.85#  [-7.770, -4.85 , -5.71] #float(input("Enter height offset: "))
# height_offset = [0,0,0]
job_no_offset = 3
point_of_rotation = np.array(
        (slicing_meta["point_of_rotation"], slicing_meta["baselayer_thickness"]))
base_thickness = slicing_meta["baselayer_thickness"]
layer_angle = np.array((slicing_meta["layer_angle"]))
print(layer_angle)
num_layer_start = 1
num_layer_end = 80
heights_all = []
flames_all = []
rms_err_all = []
flames = []
heights = []


record_folder = 'ER4043_bent_tube_small_2024_10_21_13_25_58'
recorded_dir = f'../../../../recorded_data/{record_folder}/'
height_offset = -7.81
for layer in range(num_layer_start, num_layer_end+1):
    print(f"Starting layer {layer}", end='\r')
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
    to_flat_angle = np.deg2rad(layer_angle * (layer))
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
        if flame_3d.shape[0] == 0:
            raise ValueError("No flame detected")
    except ValueError as e:
        print(e)
        flame_3d= None
    except FileNotFoundError as e:
        print(e)
        flame_3d = None
    else:
        # ammend job numbers to front
        job_no = job_no.reshape((-1,1))
        output_array = np.hstack((job_no, flame_3d))
        flames.append(output_array)
with open(f"{record_folder}_flame.pkl", 'wb') as file:
    pickle.dump(flames, file)
