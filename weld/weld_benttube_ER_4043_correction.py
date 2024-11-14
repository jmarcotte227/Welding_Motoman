import sys
import glob
import yaml
import numpy as np

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


# Setup Optimization Problem
def v_opt(v_next, v_prev, h_err, h_targ, model, beta=0.11):
    return (
        norm(h_targ + h_err - model.v2dh(v_next), 2) ** 2
        + beta * norm(delta_v(v_next), 2) ** 2
    )


bounds = Bounds(3, 17)


#####################SENSORS############################################
# weld state logging
weld_ser = RRN.SubscribeService('rr+tcp://192.168.55.10:60823?service=welder')
cam_ser = RRN.ConnectService("rr+tcp://localhost:60827/?service=camera")
# mic_ser = RRN.ConnectService('rr+tcp://localhost:60828?service=microphone')
## RR sensor objects
rr_sensors = WeldRRSensor(
    weld_service=weld_ser, cam_service=cam_ser, microphone_service=None
)

config_dir = "../config/"
flir_intrinsic = yaml.load(open(config_dir + "FLIR_A320.yaml"), Loader=yaml.FullLoader)

################################ Data Directories ###########################
now = datetime.now()

dataset = "bent_tube/"
sliced_alg = "slice_ER_4043_large_hot/"
data_dir = "../data/" + dataset + sliced_alg
rec_folder = input("Enter folder of desired test directory (leave blank for new): ")
if rec_folder == "":
    recorded_dir = now.strftime(
        "../../recorded_data/ER4043_bent_tube_large_hot_OL_%Y_%m_%d_%H_%M_%S/"
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
    tool_file_path="../config/flir_imaging.csv",
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
num_layer_start = int(0)
num_layer_end = int(1)

for layer in range(num_layer_start, num_layer_end):
    mp = MotionProgram(
        ROBOT_CHOICE="RB1",
        ROBOT_CHOICE2="ST1",
        pulse2deg=robot.pulse2deg,
        pulse2deg_2=positioner.pulse2deg,
        tool_num=12,
    )
    curve_sliced_js = np.loadtxt(
        data_dir + f"curve_sliced_js/MA2010_js{layer}_0.csv", delimiter=","
    ).reshape((-1, 6))

    positioner_js = np.loadtxt(
        data_dir + f"curve_sliced_js/D500B_js{layer}_0.csv", delimiter=","
    )
    curve_sliced_relative = np.loadtxt(
        data_dir + f"curve_sliced_relative/slice{layer}_0.csv", delimiter=","
    )

    # Define breakpoints; redundant here
    num_points_layer = len(curve_sliced_js)
    breakpoints = np.linspace(0, len(curve_sliced_js) - 1, num=num_points_layer).astype(
        int
    )

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

    q1_all = [curve_sliced_js[breakpoints[0]]]
    q2_all = [positioner_js[breakpoints[0]]]
    v1_all = [jog_vd]
    v2_all = [pos_vel]
    primitives = ["movej"]
    for j in range(1, len(breakpoints)):
        q1_all.append(curve_sliced_js[breakpoints[j]])
        q2_all.append(positioner_js[breakpoints[j]])
        v1_all.append(max(base_vd, 0.1))
        v2_all.append(pos_vel)
        primitives.append("movel")

    q_prev = positioner_js[breakpoints[-1]]
    rr_sensors.start_all_sensors()
    global_ts, timestamp_robot, joint_recording, job_line, _ = ws.weld_segment_dual(
        primitives,
        robot,
        positioner,
        q1_all,
        q2_all,
        v1_all,
        v2_all,
        cond_all=[int(base_feedrate_cmd / 10) + job_offset],
        arc=True,
        blocking=True,
    )
    rr_sensors.stop_all_sensors()
    global_ts = np.reshape(global_ts, (-1, 1))
    job_line = np.reshape(job_line, (-1, 1))

    # Reposition arm out of the way
    q_0 = client.getJointAnglesMH(robot.pulse2deg)
    q_0[1] = q_0[1] - np.pi / 8
    ws.jog_single(robot, q_0, 4)

    model = SpeedHeightModel(a=-0.36997977, b=1.21532975)
    # model = SpeedHeightModel()

    # # save data
    save_path = recorded_dir + f"layer_{layer}/"
    try:
        os.makedirs(save_path)
    except Exception as e:
        print(e)
    np.savetxt(
        save_path + "weld_js_exe.csv",
        np.hstack((global_ts, job_line, joint_recording)),
        delimiter=",",
    )
    np.savetxt(save_path + "/coeff_mat.csv", model.coeff_mat, delimiter=",")
    np.savetxt(save_path + "/model_p.csv", model.p, delimiter=",")
    rr_sensors.save_all_sensors(save_path)
    input("-------Base Layer Finished-------")

    ## Interpret base layer IR data to get h offset
    flame_3d, torch_path, job_no = flame_tracking(
        save_path, robot, robot2, positioner, flir_intrinsic
    )

    base_thickness = float(input("Enter base thickness: "))
    for i in range(flame_3d.shape[0]):
        flame_3d[i] = R.T @ flame_3d[i]
    if flame_3d.shape[0] == 0:
        height_offset = 6  # this is arbitrary
    else:
        avg_base_height = np.mean(flame_3d[:, 2])
        height_offset = base_thickness - avg_base_height

try:
    print("Average Base Height:", avg_base_height)
    print("Height Offset:", height_offset)
except:
    height_offset = float(input("Enter height offset: ")) # -8.9564 -9.1457

###########################################layer welding############################################
print("----------Normal Layers-----------")
num_layer_start = 1  ###modify layer num here
num_layer_end = 152
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
        model = SpeedHeightModel(a=-0.36997977, b=1.21532975)
        # model = SpeedHeightModel()
        vel_nom = model.dh2v(height_profile)
        velocity_profile = vel_nom
    else: 
        start_dir = not np.loadtxt(f"{recorded_dir}layer_{layer-1}/start_dir.csv", delimiter=",")
        # Initialize model with previous layer's coefficients
        # model_coeff = np.loadtxt(f"{recorded_dir}layer_{layer-1}/coeff_mat.csv", delimiter=",")
        # model_p = np.loadtxt(f"{recorded_dir}layer_{layer-1}/model_p.csv", delimiter=",")
        # model = SpeedHeightModel(a = model_coeff[0], b = model_coeff[1], p = model_p)
        model = SpeedHeightModel(a=-0.36997977, b=1.21532975)
        # model = SpeedHeightModel()
        vel_nom = model.dh2v(height_profile) #updated later down if model updates

        
        ir_error_flag = False
        ### Process IR data prev 
        try:
            flame_3d_prev, _, job_no_prev = flame_tracking(f"{recorded_dir}layer_{layer-1}/", robot, robot2, positioner, flir_intrinsic, height_offset)
            if flame_3d_prev.shape[0] == 0:
                raise ValueError("No flame detected")
        except ValueError as e:
            print(e)
            flame_3d_prev = None
            ir_error_flag = True
        else:
            # rotate to flat
            for i in range(flame_3d_prev.shape[0]):
                flame_3d_prev[i] = R.T @ flame_3d_prev[i] 
            
            new_x, new_z = rotate(
                point_of_rotation, (flame_3d_prev[:, 0], flame_3d_prev[:, 2]), to_flat_angle
            )
            flame_3d_prev[:, 0] = new_x
            flame_3d_prev[:, 2] = new_z - base_thickness

            job_no_prev = [i - job_no_offset for i in job_no_prev]
            averages_prev = avg_by_line(job_no_prev, flame_3d_prev, np.linspace(0,len(curve_sliced_js)-1,len(curve_sliced_js)))
            heights_prev = averages_prev[:,2]
            if start_dir: heights_prev = np.flip(heights_prev)
            # Find Valid datapoints for height correction
            prev_idx = np.argwhere(np.invert(np.isnan(heights_prev)))

            ### Process IR data 2 prev
            try:
                flame_3d_prev_2, _, job_no_prev_2 = flame_tracking(
                        f"{recorded_dir}layer_{layer-2}/",
                        robot,
                        robot2,
                        positioner,
                        flir_intrinsic,
                        height_offset
                )
                print(flame_3d_prev_2.shape)
            except ValueError as e:
                print(e)
                ir_error_flag = True
            else:
                print(ir_error_flag)
                # rotate to flat
                for i in range(flame_3d_prev_2.shape[0]):
                    flame_3d_prev_2[i] = R.T @ flame_3d_prev_2[i] 
                
                new_x, new_z = rotate(
                    point_of_rotation, 
                    (flame_3d_prev_2[:, 0], flame_3d_prev_2[:, 2]),
                    to_flat_angle
                )
                flame_3d_prev_2[:, 0] = new_x
                flame_3d_prev_2[:, 2] = new_z - base_thickness

                job_no_prev_2 = [i - job_no_offset for i in job_no_prev_2]
                averages_prev_2 = avg_by_line(job_no_prev_2, flame_3d_prev_2, np.linspace(0,len(curve_sliced_js)-1,len(curve_sliced_js)))
               
                heights_prev_2 = averages_prev_2[:,2]
                if not start_dir: heights_prev_2 = np.flip(heights_prev_2)
                # Find Valid datapoints for height correction
                prev_idx_2 = np.argwhere(np.invert(np.isnan(heights_prev_2)))

                # Calculate Cartesian Velocity
                calc_vel, job_nos_vel, _ = calc_velocity(f"{recorded_dir}layer_{layer-1}/",robot)
                job_nos_vel = [i - job_no_offset for i in job_nos_vel]
                vel_avg = avg_by_line(job_nos_vel, calc_vel, np.linspace(0,len(curve_sliced_js)-1, len(curve_sliced_js))).reshape(-1)
                
                # correct direction if start dir is in the opposite direction
                if start_dir:
                    vel_avg = np.flip(vel_avg)
                vel_valid_idx = np.argwhere(np.invert(np.isnan(vel_avg)))
                
                valid_idx = np.intersect1d(np.intersect1d(prev_idx, prev_idx_2), vel_valid_idx)
                dh = heights_prev[valid_idx]-heights_prev_2[valid_idx]
                print(dh)
                # update model coefficients
                # print("Update, vel_avg: ", vel_avg[valid_idx]) print("Update, dh: ", dh)
                # model.model_update_rls(vel_avg[valid_idx], dh)
                vel_nom = model.dh2v(height_profile)
                print(vel_nom)
                # if np.any(np.isnan(vel_nom)):
                #     print("bum model")
                #     model_coeff = np.loadtxt(f"{recorded_dir}layer_{layer-2}/coeff_mat.csv", delimiter=",")
                #     model_p = np.loadtxt(f"{recorded_dir}layer_{layer-2}/model_p.csv", delimiter=",")
                #     model = SpeedHeightModel(a = model_coeff[0], b = model_coeff[1], p = model_p)
                #     vel_nom = model.dh2v(height_profile)
            heights_prev = interpolate_heights(height_profile, heights_prev)
            height_err = 0-heights_prev
            # plt.plot(heights_prev)
            # plt.plot(heights_prev_2)
            # plt.show()
            # plt.close()
            # ax = plt.figure().add_subplot(projection='3d')
            # ax.plot3D(flame_3d_prev[:,0], flame_3d_prev[:,1], flame_3d_prev[:,2])
            # ax.plot3D(flame_3d_prev_2[:,0], flame_3d_prev_2[:,1], flame_3d_prev_2[:,2])
            # plt.show()
            # plt.close()

            nan_vel_idx = np.argwhere(np.isnan(vel_avg))
            vel_avg[nan_vel_idx] = vel_nom[nan_vel_idx]
            opt_result = minimize(
                v_opt,
                vel_nom,
                (vel_avg, height_err, height_profile, model),
                bounds=bounds,
                options={"maxfun": 100000},
            )
            try:
                if not opt_result.success:
                    print(opt_result)
                    raise ValueError(opt_result.message)

                velocity_profile = opt_result.x

            except ValueError as e:
                
                velocity_profile = vel_nom
            velocity_profile= vel_nom # ADDING FOR OPEN LOOP
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
    try: print("Prev velocities: ", vel_avg)
    except: pass
    print("Velocities: ",velocity_profile)
    print("Sent Velocities: ", velocity_profile[breakpoints])
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
        arc=True,
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
    print(f"-------Layer {layer} Finished-------")


