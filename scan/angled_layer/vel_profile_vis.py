import matplotlib.pyplot as plt
import numpy as np
layer = 72
fig,ax=plt.subplots(1,1)
fig.set_size_inches(10,6)
fig.set_dpi(200)
cap_data=f"../../../recorded_data/ER4043_bent_tube_2024_09_04_12_23_40/layer_{layer}/"
vel_profile = np.loadtxt(
    cap_data+"velocity_profile.csv",
    delimiter = ','
)

import sys
import yaml
import matplotlib.pyplot as plt
import numpy as np
from motoman_def import robot_obj, positioner_obj
from robotics_utils import H_inv
import scipy
import matplotlib.animation as animation
import pickle

sys.path.append('../../toolbox')
from angled_layers import rotate, flame_tracking, avg_by_line, calc_velocity, SpeedHeightModel

config_dir = "../../config/"
flir_intrinsic = yaml.load(open(config_dir + "FLIR_A320_legacy.yaml"), Loader=yaml.FullLoader)
dataset = "bent_tube/"
sliced_alg = "slice_ER_4043/"
data_dir = "../../data/" + dataset + sliced_alg

with open(data_dir + "slicing.yml", "r") as file:
    slicing_meta = yaml.safe_load(file)

robot = robot_obj(
    "MA2010_A0",
    def_path=config_dir+"MA2010_A0_robot_default_config.yml",
    tool_file_path=config_dir+"torch_no_fujimount.csv",
    pulse2deg_file_path=config_dir+"MA2010_A0_pulse2deg_real.csv",
    d=15,
)
robot2 = robot_obj(
    "MA1440_A0",
    def_path=config_dir+"MA1440_A0_robot_default_config.yml",
    tool_file_path=config_dir+"flir.csv",
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

ax.plot(vel_profile, 'r')
ax.set_xlabel("Segment Number")
ax.set_ylabel("Velocity (mm/s)")
ax.set_title(f"Layer {layer} Velocity Profile")
fig.savefig(f'velocity_profile_{layer}.png')
plt.show()
