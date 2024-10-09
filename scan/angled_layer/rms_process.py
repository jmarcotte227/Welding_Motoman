import pickle
import matplotlib.pyplot as plt
import yaml
import sys
import numpy as np
from motoman_def import robot_obj, positioner_obj
from robotics_utils import H_inv
sys.path.append('../../toolbox')
from angled_layers import avg_by_line, rotate

plt.rcParams['text.usetex'] = True

def rms_error(data):
    data = np.array(data)
    n = 0
    num = 0
    for i in data:
        if not np.isnan(i): 
            num = num + i**2
            n+=1
    return np.sqrt(num/n)

config_dir = "../../config/"
dataset = "bent_tube/"
sliced_alg = "slice_ER_4043/"
data_dir = "../../data/" + dataset + sliced_alg

# flame_set = 'processing_data/ER4043_bent_tube_2024_09_04_12_23_40_flame.pkl'
flame_sets = [
    'processing_data/ER4043_bent_tube_2024_08_28_12_24_30_flame.pkl',
    'processing_data/ER4043_bent_tube_2024_09_04_12_23_40_flame.pkl',
    # 'processing_data/ER4043_bent_tube_2024_09_03_13_26_16_flame.pkl',
]
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

H2010_1440 = H_inv(robot2.base_H)
H = np.loadtxt(data_dir + "curve_pose.csv", delimiter=",")
p = H[:3, -1]
R = H[:3, :3]

fig_2, ax_2 = plt.subplots(1,1)
fig_2.set_size_inches(10,6)
fig_2.set_dpi(200)

fig, ax= plt.subplots(1,1)
fig.set_size_inches(10,6)
fig.set_dpi(200)
plt_params = [
    'b--',
    'r',
    'g-.'
]

layer_start = 1
rms_errs = []

for idx,flame_set in enumerate(flame_sets):
    with open(flame_set, 'rb') as file:
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

    height_profile = []
    for distance in dist_to_por:
        height_profile.append(distance * np.sin(np.deg2rad(layer_angle)))
    height_err = []
    flames_flat = []
    # for layer, flame in enumerate(flames):
    for layer, flame in enumerate(flames):
        to_flat_angle = np.deg2rad(layer_angle*(layer+layer_start))
        for i in range(flame.shape[0]):
            flame[i,1:] = R.T @ flame[i,1:]

        new_x, new_z = rotate(
            point_of_rotation, (flame[:, 1], flame[:, 3]), to_flat_angle
        )
        flame[:, 1] = new_x
        flame[:, 3] = new_z - base_thickness

        print(flame[:,0])
        flame[:,0] = flame[:,0]-job_no_offset
        averages= avg_by_line(flame[:,0], flame[:,1:], np.linspace(0,49,50))
        print(averages)
        height_err.append(averages[:,2])
        flames_flat.append(averages)
        # if idx == 1:
        #     ax.plot(np.linspace(1,50,50),averages[:,2], plt_params[idx])
        ax.plot(np.linspace(1,50,50),averages[:,2], plt_params[idx])
    
        


    # fig = plt.figure()
    # ax = plt.axes(projection = '3d')
    # for layer in flames_flat:
    #     ax.scatter(layer[:,0],layer[:,1],layer[:,2])
    # fig.show()

    rms_err = []
    for scan in height_err:
        rms_err.append(rms_error(scan))
    rms_errs.append(rms_err)
# ax.plot(np.linspace(1,50,50),height_profile, plt_params[2])
# ax.plot(np.linspace(1,50,50),np.zeros(50), plt_params[2])
print(len(height_err))
print("max openloop error: ", np.max(rms_errs[0]))
print("max closed-loop error: ", np.max(rms_errs[1]))
ax_2.plot(np.linspace(1,len(height_err),len(height_err)),rms_errs[0], plt_params[0])
ax_2.plot(np.linspace(1,len(height_err),len(height_err)),rms_errs[1], plt_params[1])
ax_2.set_xlabel("Layer Number")
ax_2.set_ylabel("RMSE (mm)")
ax_2.legend(["Open-Loop", "Closed-Loop"])
ax.legend(["$e_{16}$ Open-Loop", "$e_{16}$ Closed-Loop"])
# ax.legend(["$h_{16}$ Open-Loop","$h_{16}$ Closed-Loop", "$\delta h_{d,16}$"])
# ax.legend(["$h_{72}$ Closed-Loop", "$\delta h_{d,72}$"])
ax.set_title(f"Height Profile Layer {layer_start+1}")
ax.set_xlabel("Segment Index")
ax.set_ylabel("Height (mm)")
# ax.set_ylabel("Error (mm)")
ax.set_ylim(-7,3)
ax.grid()
ax_2.grid()
fig_2.savefig('rms_plot.png')
fig.savefig(f'error_profile_{layer_start+1}.png')
plt.show()


