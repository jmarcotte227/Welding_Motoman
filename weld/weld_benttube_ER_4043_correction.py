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
from angled_layer import *


# Setup Optimization Problem
def v_opt(v_next, v_prev, h_err, h_targ, model, alpha=0.04, beta=0.11):
    return (
        norm(h_targ + h_err - model.v2dh(v_next), 2) ** 2
        + alpha * norm(v_next - v_prev, 2) ** 2
        + beta * norm(delta_v(v_next), 2) ** 2
    )


bounds = Bounds(3, 17)


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
rec_folder = input("Enter folder of desired test directory (leave blank for new): ")
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
    )
    rr_sensors.stop_all_sensors()
    global_ts = np.reshape(global_ts, (-1, 1))
    job_line = np.reshape(job_line, (-1, 1))

    # Reposition arm out of the way
    q_0 = client.getJointAnglesMH(robot.pulse2deg)
    q_0[1] = q_0[1] - np.pi / 8
    ws.jog_single(robot, q_0, 4)

    # save data
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
    rr_sensors.save_all_sensors(save_path)
    input("-------Base Layer Finished-------")

    ## Interpret base layer IR data to get h offset
    flame_3d, torch_path, job_no = flame_tracking(save_path, robot, robot2, positioner,flir_intrinsic) 
    
    base_thickness = float(input("Enter base thickness: "))
    for i in range(flame_3d.shape[0]):
      flame_3d[i] = R.T@flame_3d[i]
    if flame_3d.shape[0] == 0:
      height_offset = 6 # this is arbitrary
    else:
      avg_base_height = np.mean(flame_3d[:,2])
      height_offset = base_thickness - avg_base_height

try:
  print("Height Offset:", height_offset)
except:
  height_offset = float(input("Enter height offset: "))




###########################################layer welding############################################
print("----------Normal Layers-----------")
num_layer_start = 1  ###modify layer num here
num_layer_end = 80

q_prev=client.getJointAnglesDB(positioner.pulse2deg)
# q_prev = np.array([9.53e-02, -2.71e00])  ###for motosim tests only

base_thickness = slicing_meta["baselayer_thickness"]
print("start layer: ", num_layer_start)
print("end layer: ", num_layer_end)
layer_angle = np.array((slicing_meta["layer_angle"]))

start_dir = True  # Alternate direction that the layer starts from

# for layer in range(num_layer_start,num_layer_end,nominal_slice_increment):
for layer in range(num_layer_start, num_layer_end):
    if layer != 1:
        save_path = recorded_dir + "layer_" + str(layer - 1) + "/"
        start_dir_prev = np.loadtxt(save_path + "start_dir.csv", delimiter=",")
        start_dir = not start_dir_prev
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
    curve_sliced = np.loadtxt(
        data_dir + f"curve_sliced/slice{layer}_0.csv", delimiter=","
    )

    ###alternate the start point on different ends
    num_points_layer = len(curve_sliced_js)
    if start_dir:
        breakpoints = np.linspace(
            0, len(curve_sliced_js) - 1, num=num_points_layer
        ).astype(int)
    else:
        breakpoints = np.linspace(
            len(curve_sliced_js) - 1, 0, num=num_points_layer
        ).astype(int)

    # jog to start and position camera
    p_positioner_home = np.mean(
        [robot.fwd(curve_sliced_js[0]).p, robot.fwd(curve_sliced_js[-1]).p], axis=0
    )
    p_robot2_proj = p_positioner_home + np.array([0, 0, 50])
    p2_in_base_frame = np.dot(H2010_1440[:3, :3], p_robot2_proj) + H2010_1440[:3, 3]
    v_z = H2010_1440[:3, :3] @ np.array(
        [0, -0.96592582628, -0.2588190451]
    )  ###pointing toward positioner's X with 15deg tiltd angle looking down
    v_y = VectorPlaneProjection(
        np.array([-1, 0, 0]), v_z
    )  ###FLIR's Y pointing toward 1440's -X in 1440's base frame, projected on v_z's plane
    v_x = np.cross(v_y, v_z)
    p2_in_base_frame = (
        p2_in_base_frame - measure_distance * v_z
    )  ###back project measure_distance-mm away from torch
    R2 = np.vstack((v_x, v_y, v_z)).T
    q2 = robot2.inv(p2_in_base_frame, R2, last_joints=np.zeros(6))[0]
    q_prev = client.getJointAnglesDB(positioner.pulse2deg)
    num2p = np.round((q_prev - positioner_js[0]) / (2 * np.pi))
    positioner_js += num2p * 2 * np.pi
    ws.jog_dual(robot2, positioner, q2, positioner_js[0], v=1)

    ###########################################velocity profile#########################################
    dh_max = slicing_meta["dh_max"]
    dh_min = slicing_meta["dh_min"]
    point_of_rotation = np.array(
        (slicing_meta["point_of_rotation"], slicing_meta["baselayer_thickness"])
    )
    ##calculate distance to point of rotation
    dist_to_por = []
    for i in range(len(curve_sliced)):
        point = np.array((curve_sliced[i, 0], curve_sliced[i, 2]))
        dist = np.linalg.norm(point - point_of_rotation)
        dist_to_por.append(dist)

    height_profile = []
    for distance in dist_to_por:
        height_profile.append(distance * np.sin(np.deg2rad(layer_angle)))
    vel_nom = weld_dh2v.dh2v_loglog(height_profile, feedrate_cmd, "ER_4043")

    ################################# Check for correction data, and plan velocity accordingly #############################
    if layer == 1:
        velocity_profile = vel_nom
        model = speedHeightModel()
        save_path = recorded_dir + "layer_" + str(layer) + "/"
        try:
            os.makedirs(save_path)
        except Exception as e:
            print(e)
        np.savetxt(save_path + "/model_coeff.csv", model.coeff_mat, delimiter=",")
    else:
        save_path = recorded_dir + "layer_" + str(layer - 1) + "/"

        # get previous model coefficients
        model_coeff = np.loadtxt(save_path + "model_coeff.csv", delimiter=",")
        model = speedHeightModel(coeff_mat=model_coeff)
        # load previous velocity data
        vel_prev = np.loadtxt(save_path + "velocity_profile.csv", delimiter=",")

        # Process IR data
        with open(save_path + "ir_recording.pickle", "rb") as file:
            ir_recording = pickle.load(file)
        ir_ts = np.loadtxt(save_path + "ir_stamps.csv", delimiter=",")
        joint_angle = np.loadtxt(save_path + "weld_js_exe.csv", delimiter=",")
        if len(ir_ts) == 0:
            print("No Flame Detected, nominal velocity")
            velocity_profile = vel_nom
        else:
            timeslot = [ir_ts[0] - ir_ts[0], ir_ts[-1] - ir_ts[0]]
            duration = np.mean(np.diff(timeslot))

            flame_3d = []
            job_no = []
            torch_path = []
            for start_time in timeslot[:-1]:

                start_idx = np.argmin(np.abs(ir_ts - ir_ts[0] - start_time))
                end_idx = np.argmin(np.abs(ir_ts - ir_ts[0] - start_time - duration))

                # find all pixel regions to record from flame detection
                for i in range(start_idx, end_idx):

                    ir_image = ir_recording[i]
                    try:
                        centroid, bbox = flame_detection_aluminum(
                            ir_image, percentage_threshold=0.8
                        )
                    except ValueError:
                        centroid = None
                    if centroid is not None:
                        # find spatial vector ray from camera sensor
                        vector = np.array(
                            [
                                (centroid[0] - flir_intrinsic["c0"])
                                / flir_intrinsic["fsx"],
                                (centroid[1] - flir_intrinsic["r0"])
                                / flir_intrinsic["fsy"],
                                1,
                            ]
                        )
                        vector = vector / np.linalg.norm(vector)
                        # find index closest in time of joint_angle
                        joint_idx = np.argmin(np.abs(ir_ts[i] - joint_angle[:, 0]))
                        robot2_pose_world = robot2.fwd(
                            joint_angle[joint_idx][8:-2], world=True
                        )
                        p2 = robot2_pose_world.p
                        v2 = robot2_pose_world.R @ vector
                        robot1_pose = robot.fwd(joint_angle[joint_idx][2:8])
                        p1 = robot1_pose.p
                        v1 = robot1_pose.R[:, 2]
                        positioner_pose = positioner.fwd(
                            joint_angle[joint_idx][-2:], world=True
                        )

                        # find intersection point
                        intersection = line_intersect(p1, v1, p2, v2)
                        intersection = positioner_pose.R.T @ (
                            intersection - positioner_pose.p
                        )
                        torch = positioner_pose.R.T @ (
                            robot1_pose.p - positioner_pose.p
                        )

                        flame_3d.append(intersection)
                        torch_path.append(intersection)
                        job_no.append(int(joint_angle[joint_idx][1]))
            flame_3d = np.array(flame_3d)
            torch_path = np.array(torch_path)
            with open(
                save_path + "flame_3d_" + str(layer - 1) + "_0.pkl", "wb"
            ) as file:
                pickle.dump(flame_3d, file)
            with open(
                save_path + "torch_path_" + str(layer - 1) + "_0.pkl", "wb"
            ) as file:
                pickle.dump(flame_3d, file)
            with open(save_path + "ir_job_no_" + str(layer - 1) + ".pkl", "wb") as file:
                pickle.dump(job_no, file)

            ###################### Plot Flame vs Anticipated Layers ############################
            print("Flame Processed | Plotting Now")
            print(flame_3d.shape)
            # account for offset in height
            try:
                flame_3d[:, 2] = flame_3d[:, 2] + height_offset
            except:
                pass
            # plot the flame 3d
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            try:
                ax.scatter(flame_3d[:, 0], flame_3d[:, 1], flame_3d[:, 2], "b")
                # ax.plot3D(torch_path[:,0], torch_path[:,1], torch_path[:,2])
            except IndexError:
                print("No Flame Detected")

            curve_sliced_relative = np.loadtxt(
                data_dir
                + "curve_sliced_relative/slice"
                + str(layer)
                + "_"
                + str(x)
                + ".csv",
                delimiter=",",
            )
            ax.plot3D(
                curve_sliced_relative[:, 0],
                curve_sliced_relative[:, 1],
                curve_sliced_relative[:, 2],
            )
            ax.set_aspect("equal")
            plt.show()
            plt.close()

            #################### Get Height Profile ######################
            for i in range(flame_3d.shape[0]):
                flame_3d[i] = R.T @ flame_3d[i]
            # try:
            #   print(flame_3d[:,2])
            # except:
            #   pass
            print(np.any(flame_3d[:, 2]).any() > 400)
            if np.greater(flame_3d, 400).any():
                print("Flame out of bounds Error")
                velocity_profile = vel_nom
            else:
                to_flat_angle = np.deg2rad(layer_angle * (layer - 1))
                new_x, new_z = rotate(
                    point_of_rotation, (flame_3d[:, 0], flame_3d[:, 2]), to_flat_angle
                )
                flame_3d[:, 0] = new_x
                flame_3d[:, 2] = new_z - base_thickness

                job_no_offset = (
                    3  # get rid of first non-instructive job lines in AAA file
                )
                job_no = [i - job_no_offset for i in job_no]
                averages = avg_by_line(job_no, flame_3d, len(breakpoints))
                print("averages: ", averages)
                scan_height = interpolate_heights(
                    averages[:, -1]
                )  # checks for empty height quantities and fills them in
                height_err = 0 - scan_height
                if layer > 2:
                    try:
                        ######### calculate previous height profile ##########
                        with open(
                            recorded_dir
                            + "layer_"
                            + str(layer - 2)
                            + "/flame_3d_"
                            + str(layer - 2)
                            + "_0.pkl",
                            "rb",
                        ) as file:
                            flame_3d_prev = pickle.load(file)
                        with open(
                            recorded_dir
                            + "layer_"
                            + str(layer - 2)
                            + "/ir_job_no_"
                            + str(layer - 2)
                            + ".pkl",
                            "rb",
                        ) as file:
                            job_no_prev = pickle.load(file)
                    except:
                        flame_3d_prev = np.array([])
                        job_no_prev = np.array([])

                    job_no_prev = [i - job_no_offset for i in job_no_prev]
                    # account for offset in height
                    try:
                        flame_3d_prev[:, 2] = flame_3d_prev[:, 2] + height_offset
                    except:
                        pass
                    if flame_3d_prev.shape[0] > 1:
                        for i in range(flame_3d_prev.shape[0]):
                            flame_3d_prev[i] = R.T @ flame_3d_prev[i]
                        layer_angle = layer_angle * (layer)
                        layer_angle = np.deg2rad(layer_angle)
                        # print(flame_3d.shape)
                        new_x, new_z = rotate(
                            point_of_rotation,
                            (flame_3d_prev[:, 0], flame_3d_prev[:, 2]),
                            layer_angle,
                        )
                        flame_3d_prev[:, 0] = new_x
                        flame_3d_prev[:, 2] = new_z - base_thickness
                        print(len(job_no_prev))
                        print(flame_3d_prev.shape)
                        averages = avg_by_line(job_no_prev, flame_3d_prev, 50)
                        heights_prev = interpolate_heights(averages[:, -1])
                        if not start_dir_prev:
                            heights_prev = heights_prev[::-1]

                    ######## update model ###########
                    try:
                        joint_idx_start = np.where(joint_angle[:, 1] == job_no_offset)[
                            0
                        ][0]
                    except:
                        joint_idx_start = np.where(
                            joint_angle[:, 1] == job_no_offset + 1
                        )[0][0]

                    joint_angles_prev = joint_angle[joint_idx_start - 1, :]
                    joint_angles_layer = joint_angle[joint_idx_start:, :]
                    pose_prev = robot.fwd(joint_angles_prev[2:8]).p
                    time_prev = joint_angles_prev[0]

                    time_series_layer = []
                    job_nos_layer = []
                    calc_vel_layer = []
                    for i in range(joint_angles_layer.shape[0]):
                        # calculate instantatneous velocity
                        robot1_pose = robot.fwd(joint_angles_layer[i, 2:8])
                        cart_dif = robot1_pose.p - pose_prev
                        time_stamp = joint_angles_layer[i][0]
                        time_dif = time_stamp - time_prev
                        time_prev = time_stamp
                        cart_vel = cart_dif / time_dif
                        time_prev = time_stamp
                        pose_prev = robot1_pose.p
                        lin_vel = np.sqrt(
                            cart_vel[0] ** 2 + cart_vel[1] ** 2 + cart_vel[2] ** 2
                        )
                        calc_vel_layer.append(lin_vel)
                        job_nos_layer.append(joint_angles_layer[i][1])
                        time_series_layer.append(time_stamp)

                    job_nos_layer = [i - job_no_offset for i in job_nos_layer]
                    vel_avg = avg_by_line_vel(job_nos_layer, calc_vel_layer, 50)
                    # print(scan_height)
                    # print(heights_prev)
                    try:
                        dh = scan_height - heights_prev
                        print("Old Model Coeffs: ", model.coeff_mat)
                        model.model_update_RLS(vel_avg[1:-2], dh[1:-2])
                        print("New Model Coeffs: ", model.coeff_mat)
                    except Exception as e:
                        print(e)
                try:
                    print(vel_prev)
                    print(dh)
                    plt.plot(scan_height)
                    plt.plot(heights_prev)
                    plt.show()
                    plt.close()
                    plt.scatter(vel_prev, dh)
                    plt.plot(vel_prev, model.v2dh(vel_prev))
                    plt.show()
                    plt.close()
                except:
                    pass

                opt_result = minimize(
                    v_opt,
                    vel_nom,
                    (vel_prev, height_err, height_profile, model),
                    bounds=bounds,
                    options={"maxfun": 100000},
                )

                if not opt_result.success:
                    print(opt_result)
                    raise ValueError(opt_result.message)

                velocity_profile = opt_result.x
                if start_dir_prev == False:
                    velocity_profile = np.flip(velocity_profile)
            ###########################################################################

    print(velocity_profile[breakpoints])
    print("Start dir: ", start_dir)
    if layer != 1:
        print("Prev start dir: ", start_dir_prev)
    input("Check Vel Profile, enter to continue")
    print("Breakpoints Length: ", len(breakpoints))

    save_path = recorded_dir + "layer_" + str(layer) + "/"
    try:
        os.makedirs(save_path)
    except Exception as e:
        print(e)
    # velocity_profile = np.flip(velocity_profile)
    np.savetxt(save_path + "/model_coeff.csv", model.coeff_mat, delimiter=",")
    np.savetxt(
        save_path + "velocity_profile.csv", velocity_profile[breakpoints], delimiter=","
    )
    np.savetxt(save_path + "start_dir.csv", [start_dir], delimiter=",")
    ###move to intermidieate waypoint for collision avoidance if multiple section
    if num_sections != num_sections_prev:
        waypoint_pose = robot.fwd(curve_sliced_js[breakpoints[0]])
        waypoint_pose.p[-1] += 50
        q1 = robot.inv(
            waypoint_pose.p, waypoint_pose.R, curve_sliced_js[breakpoints[0]]
        )[0]
        q2 = positioner_js[breakpoints[0]]
        ws.jog_dual(robot, positioner, q1, q2, v=100)
    q1_all = [curve_sliced_js[breakpoints[0]]]
    q2_all = [positioner_js[breakpoints[0]]]
    v1_all = [4]
    v2_all = [10]
    primitives = ["movej"]
    for j in range(0, len(breakpoints)):
        q1_all.append(curve_sliced_js[breakpoints[j]])
        q2_all.append(positioner_js[breakpoints[j]])
        v1_all.append(max(velocity_profile[breakpoints[j]], 0.1))

        positioner_w = vd_relative / np.linalg.norm(
            curve_sliced_relative[breakpoints[j]][:2]
        )
        v2_all.append(min(100, 100 * positioner_w / positioner.joint_vel_limit[1]))
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

    print(global_ts.shape)
    print(job_line.shape)
    print(joint_recording.shape)
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
    # input("-------Layer Finished-------")

    ################# PROCESS IR DATA #############################

    #################### Generate Corrective Action ###################################
