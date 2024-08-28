import sys
import glob
import yaml

sys.path.append("../toolbox/")
sys.path.append("")
from lambda_calc import *
from multi_robot import *
from dx200_motion_program_exec_client import *
from WeldSend import *
from RobotRaconteur.Client import *
from weldRRSensor import *
from motoman_def import *
from flir_toolbox import *
import weld_dh2v
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.optimize import minimize, Bounds
from numpy.linalg import norm
from matplotlib import pyplot as plt
from angled_layers import *

#####################SENSORS############################################
# weld state logging
# weld_ser = RRN.SubscribeService('rr+tcp://192.168.55.10:60823?service=welder')
cam_ser = RRN.ConnectService("rr+tcp://localhost:60827/?service=camera")
# mic_ser = RRN.ConnectService('rr+tcp://192.168.55.20:60828?service=microphone')
## RR sensor objects
rr_sensors = WeldRRSensor(
    weld_service=None, cam_service=cam_ser, microphone_service=None
)

config_dir = "../config/"
flir_intrinsic = yaml.load(open(config_dir + "FLIR_A320.yaml"), Loader=yaml.FullLoader)

################################ Data Directories ###########################
now = datetime.now()

dataset = "bent_tube/"
sliced_alg = "slice_ER_4043/"
data_dir = "../data/" + dataset + sliced_alg
rec_folder = 'ER4043_bent_tube_2024_08_22_11_12_27' #input("Enter folder of desired test directory (leave blank for new): ")
if rec_folder == "":
    recorded_dir = now.strftime(
        "../../recorded_data/ER4043_bent_tube_%Y_%m_%d_%H_%M_%S/"
    )
else:
    recorded_dir = "../../recorded_data/" + rec_folder + "/"
with open(data_dir + "slicing.yml", "r") as file:
    slicing_meta = yaml.safe_load(file)

robot = robot_obj(
    "MA2010_A0",
    def_path="../config/MA2010_A0_robot_default_config.yml",
    tool_file_path="../config/torch.csv",
    pulse2deg_file_path="../config/MA2010_A0_pulse2deg_real.csv",
    d=15,
)
robot2 = robot_obj(
    "MA1440_A0",
    def_path="../config/MA1440_A0_robot_default_config.yml",
    tool_file_path="../config/flir.csv",
    pulse2deg_file_path="../config/MA1440_A0_pulse2deg_real.csv",
    base_transformation_file="../config/MA1440_pose.csv",
)
positioner = positioner_obj(
    "D500B",
    def_path="../config/D500B_robot_extended_config.yml",
    tool_file_path="../config/positioner_tcp.csv",
    pulse2deg_file_path="../config/D500B_pulse2deg_real.csv",
    base_transformation_file="../config/D500B_pose.csv",
)

H2010_1440 = H_inv(robot2.base_H)

client = MotionProgramExecClient()
ws = WeldSend(client)

H = np.loadtxt(data_dir + "curve_pose.csv", delimiter=",")
p = H[:3, -1]
R = H[:3, :3]

###set up control parameters
job_offset = 200  ###200 for Aluminum ER4043
feedrate_cmd = 160
base_feedrate_cmd = 300
base_vd = 5.0
measure_distance = 500
pos_vel = 10.0
jog_vd = 4.0
job_no_offset = 3

####################BASE layer welding################################
# num_layer_start = int(0)
# num_layer_end = int(1)

# for layer in range(num_layer_start, num_layer_end):
#     mp = MotionProgram(
#         ROBOT_CHOICE="RB1",
#         ROBOT_CHOICE2="ST1",
#         pulse2deg=robot.pulse2deg,
#         pulse2deg_2=positioner.pulse2deg,
#         tool_num=12,
#     )
#     curve_sliced_js = np.loadtxt(
#         data_dir + f"curve_sliced_js/MA2010_js{layer}_0.csv", delimiter=","
#     ).reshape((-1, 6))

#     positioner_js = np.loadtxt(
#         data_dir + f"curve_sliced_js/D500B_js{layer}_0.csv", delimiter=","
#     )
#     curve_sliced_relative = np.loadtxt(
#         data_dir + f"curve_sliced_relative/slice{layer}_0.csv", delimiter=","
#     )

#     # Define breakpoints; redundant here
#     num_points_layer = len(curve_sliced_js)
#     breakpoints = np.linspace(0, len(curve_sliced_js) - 1, num=num_points_layer).astype(
#         int
#     )

#     #### jog to start and position camera
#     p_positioner_home = np.mean(
#         [robot.fwd(curve_sliced_js[0]).p, robot.fwd(curve_sliced_js[-1]).p], axis=0
#     )
#     p_robot2_proj = p_positioner_home + np.array([0, 0, 50])
#     p2_in_base_frame = np.dot(H2010_1440[:3, :3], p_robot2_proj) + H2010_1440[:3, 3]
#     # pointing toward positioner's X with 15deg tiltd angle looking down
#     v_z = H2010_1440[:3, :3] @ np.array([0, -0.96592582628, -0.2588190451])
#     # FLIR's Y pointing toward 1440's -X in 1440's base frame,
#     # projected on v_z's plane
#     v_y = VectorPlaneProjection(np.array([-1, 0, 0]), v_z)
#     v_x = np.cross(v_y, v_z)
#     # back project measure_distance-mm away from torch
#     p2_in_base_frame = p2_in_base_frame - measure_distance * v_z
#     R2 = np.vstack((v_x, v_y, v_z)).T
#     q2 = robot2.inv(p2_in_base_frame, R2, last_joints=np.zeros(6))[0]
#     q_prev = client.getJointAnglesDB(positioner.pulse2deg)
#     num2p = np.round((q_prev - positioner_js[0]) / (2 * np.pi))
#     positioner_js += num2p * 2 * np.pi
#     ws.jog_dual(robot2, positioner, q2, positioner_js[0], v=1)

#     q1_all = [curve_sliced_js[breakpoints[0]]]
#     q2_all = [positioner_js[breakpoints[0]]]
#     v1_all = [jog_vd]
#     v2_all = [pos_vel]
#     primitives = ["movej"]
#     for j in range(1, len(breakpoints)):
#         q1_all.append(curve_sliced_js[breakpoints[j]])
#         q2_all.append(positioner_js[breakpoints[j]])
#         v1_all.append(max(base_vd, 0.1))
#         v2_all.append(pos_vel)
#         primitives.append("movel")

#     q_prev = positioner_js[breakpoints[-1]]
#     rr_sensors.start_all_sensors()
#     global_ts, timestamp_robot, joint_recording, job_line, _ = ws.weld_segment_dual(
#         primitives,
#         robot,
#         positioner,
#         q1_all,
#         q2_all,
#         v1_all,
#         v2_all,
#         cond_all=[int(base_feedrate_cmd / 10) + job_offset],
#         arc=False,
#         blocking=True,
#     )
#     rr_sensors.stop_all_sensors()
#     global_ts = np.reshape(global_ts, (-1, 1))
#     job_line = np.reshape(job_line, (-1, 1))

#     # Reposition arm out of the way
#     q_0 = client.getJointAnglesMH(robot.pulse2deg)
#     q_0[1] = q_0[1] - np.pi / 8
#     ws.jog_single(robot, q_0, 4)

#     model = SpeedHeightModel()

#     # save data
#     save_path = recorded_dir + f"layer_{layer}/"
#     try:
#         os.makedirs(save_path)
#     except Exception as e:
#         print(e)
#     np.savetxt(
#         save_path + "weld_js_exe.csv",
#         np.hstack((global_ts, job_line, joint_recording)),
#         delimiter=",",
#     )
#     np.savetxt(save_path + "/coeff_mat.csv", model.coeff_mat, delimiter=",")
#     np.savetxt(save_path + "/model_p.csv", model.p, delimiter=",")
#     rr_sensors.save_all_sensors(save_path)
#     input("-------Base Layer Finished-------")

###########################################layer welding############################################
print("----------Normal Layers-----------")
num_layer_start = 1  ###modify layer num here
num_layer_end = 81
point_of_rotation = np.array(
        (slicing_meta["point_of_rotation"], slicing_meta["baselayer_thickness"])
    )
q_prev = client.getJointAnglesDB(positioner.pulse2deg)
# q_prev = np.array([9.53e-02, -2.71e00])  ###for motosim tests only

base_thickness = slicing_meta["baselayer_thickness"]
print("start layer: ", num_layer_start)
print("end layer: ", num_layer_end)
layer_angle = np.array((slicing_meta["layer_angle"]))


for layer in range(num_layer_start, num_layer_end):
    ### Initialize model
    model = SpeedHeightModel()
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

    height_profile = []
    for distance in dist_to_por:
        height_profile.append(distance * np.sin(np.deg2rad(layer_angle)))

    if layer == 1: 
        start_dir=True
        vel_nom = model.dh2v(height_profile)
        velocity_profile = vel_nom
    else: 
        start_dir = not np.loadtxt(f"{recorded_dir}layer_{layer-1}/start_dir.csv", delimiter=",")
        vel_nom = model.dh2v(height_profile) #updated later down if model updates

        
    if start_dir:
        breakpoints = np.linspace(
            0, len(curve_sliced_js) - 1, num=len(curve_sliced_js)
        ).astype(int)
    else:
        breakpoints = np.linspace(
            len(curve_sliced_js) - 1, 0, num=len(curve_sliced_js)
        ).astype(int)

    #### jog to start and position camera
    p_positioner_home = np.mean(
        [robot.fwd(curve_sliced_js[0]).p, robot.fwd(curve_sliced_js[-1]).p], axis=0
    )
    p_robot2_proj = p_positioner_home + np.array([0, 0, 50])
    p2_in_base_frame = np.dot(H2010_1440[:3, :3], p_robot2_proj) + H2010_1440[:3, 3]
    # pointing toward positioner's X with 15deg tiltd angle looking down
    v_z = H2010_1440[:3, :3] @ np.array([0, -0.96592582628, -0.2588190451])
    # FLIR's Y pointing toward 1440's -X in 1440's base frame,
    # projected on v_z's plane
    v_y = VectorPlaneProjection(np.array([-1, 0, 0]), v_z)
    v_x = np.cross(v_y, v_z)
    # back project measure_distance-mm away from torch
    p2_in_base_frame = p2_in_base_frame - measure_distance * v_z
    R2 = np.vstack((v_x, v_y, v_z)).T
    q2 = robot2.inv(p2_in_base_frame, R2, last_joints=np.zeros(6))[0]
    q_prev = client.getJointAnglesDB(positioner.pulse2deg)
    num2p = np.round((q_prev - positioner_js[0]) / (2 * np.pi))
    positioner_js += num2p * 2 * np.pi
    ws.jog_dual(robot2, positioner, q2, positioner_js[0], v=1)

    print("Start_dir: ", start_dir)
    print("Velocities: ",velocity_profile[breakpoints])
    input("Check Vel Profile, enter to continue")
    save_path = recorded_dir + "layer_" + str(layer) + "/"
    try:
        os.makedirs(save_path)
    except Exception as e:
        print(e)

    np.savetxt(save_path + "/coeff_mat.csv", model.coeff_mat, delimiter=",")
    np.savetxt(save_path + "/model_p.csv", model.p, delimiter=",")
    np.savetxt(
        save_path + "velocity_profile.csv", velocity_profile[breakpoints], delimiter=","
    )
    np.savetxt(save_path + "start_dir.csv", [start_dir], delimiter=",")
    
    q1_all = [curve_sliced_js[breakpoints[0]]]
    q2_all = [positioner_js[breakpoints[0]]]
    v1_all = [jog_vd]
    v2_all = [pos_vel]
    primitives = ["movej"]
    for j in range(0, len(breakpoints)):
        q1_all.append(curve_sliced_js[breakpoints[j]])
        q2_all.append(positioner_js[breakpoints[j]])
        v1_all.append(max(velocity_profile[breakpoints[j]], 0.1))
        v2_all.append(pos_vel)
        primitives.append("movel")
    q_prev = positioner_js[breakpoints[-1]]
    ################ Weld with sensors #############################
    rr_sensors.start_all_sensors()
    global_ts, robot_ts, joint_recording, job_line, _ = ws.weld_segment_dual(
        primitives,
        robot,
        positioner,
        q1_all,
        q2_all,
        v1_all,
        v2_all,
        cond_all=[int(feedrate_cmd / 10) + job_offset],
        arc=False,
        blocking=True,
    )
    rr_sensors.stop_all_sensors()
    global_ts = np.reshape(global_ts, (-1, 1))
    job_line = np.reshape(job_line, (-1, 1))

    # save data
    np.savetxt(
        save_path + "weld_js_exe.csv",
        np.hstack((global_ts, job_line, joint_recording)),
        delimiter=",",
    )
    rr_sensors.save_all_sensors(save_path)

    q_0 = client.getJointAnglesMH(robot.pulse2deg)
    q_0[1] = q_0[1] - np.pi / 8
    print(q_0)
    ws.jog_single(robot, q_0, 4)
    input(f"-------Layer {layer} Finished-------")


