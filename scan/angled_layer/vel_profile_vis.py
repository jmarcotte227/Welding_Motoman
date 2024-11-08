import matplotlib.pyplot as plt
import numpy as np
layer = 80
fig,ax=plt.subplots(1,1)
fig.set_size_inches(10,6)
fig.set_dpi(200)
cap_data=f"../../../recorded_data/ER4043_bent_tube_large_hot_2024_11_06_12_27_19/layer_{layer}/"
vel_profile = np.loadtxt(
    cap_data+"velocity_profile.csv",
    delimiter = ','
)
# plt.rcParams['text.usetex'] = True

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

job_no_offset=3
cart_vels = []
job_no = []
joint_angle = np.loadtxt(cap_data+'weld_js_exe.csv', delimiter=',')
for idx in range(joint_angle.shape[0]):
    robot1_pose=robot.fwd(joint_angle[idx][2:8])
    time_stamp=joint_angle[idx][0]
    if idx==0: 
        pose_prev=robot1_pose.p
        time_prev=time_stamp
    else:
        cart_dif=robot1_pose.p-pose_prev
        time_dif = time_stamp-time_prev
        time_prev = time_stamp
        cart_vel = cart_dif/time_dif
        time_prev = time_stamp
        pose_prev = robot1_pose.p
        lin_vel = np.sqrt(cart_vel[0]**2+cart_vel[1]**2)
        cart_vels.append(lin_vel)
        job_no.append(int(joint_angle[idx][1]))

job_no=[x-job_no_offset for x in job_no]
avg_vels = avg_by_line(np.array(job_no), np.array(cart_vels), np.linspace(0,49,50))
print(avg_vels.shape)
ax.plot(np.linspace(1,50,50),vel_profile, 'b--')
ax.plot(np.linspace(1,50,50),avg_vels, 'r')
ax.set_xlabel("Segment Number")
ax.set_ylabel("Velocity (mm/s)")
ax.set_title(f"Layer {layer} Velocity Profile")
ax.grid()
ax.legend(["Set Velocity", "Measured Velocity"])
fig.savefig(f'velocity_profile_{layer}.png')
plt.show()
