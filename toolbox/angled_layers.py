"""
Contains the relevant python functions and objects for welding parts
using non-uniform height profiles.
"""

import time
import pickle
import numpy as np
import cv2
from flir_toolbox import *
import matplotlib.pyplot as plt
from scipy.signal import iirfilter, sosfilt



def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy


def delta_v(v):
    """
    returns the vector containing the differences between neighboring
    elements.
    """
    delta_v_vec = [v[i] - v[i + 1] for i in range(len(v) - 1)]
    return delta_v_vec


def avg_by_line(labels, data_array, bins):
    """
    Averages the data in the variable "data" into the bins in "bins"
    according to the values in "job_line"
    """
    try:
        data_dim = data_array.shape[1]
    except IndexError:
        data_array = np.reshape(data_array, (-1,1))
        data_dim = data_array.shape[1]
    output = np.zeros((bins.shape[0], data_dim))
    for idx, _bin in enumerate(bins):
        avg_idxs = np.where(labels == _bin)[0]
        if avg_idxs.shape[0] == 0:
            output[idx, :] = [None] * data_dim
        else:
            total = np.sum(data_array[avg_idxs, :], axis=0)
            output[idx, :] = total / avg_idxs.shape[0]
    return output


class SpeedHeightModel:
    """
    Model relating dh to torch speed according to the equation
    ln(h) = a ln(v) + b
    """

    def __init__(self, lam=0.05, beta=0.99, a=-0.4619, b=1.647, p = None):
        # Beta == 1 for non-exponentail updates
        self.coeff_mat = np.array([a, b])
        self.nom_a = a
        self.nom_b = b
        self.lam = lam
        if p is None:
            self.p = np.diag(np.ones(self.coeff_mat.shape[0]) * self.lam)
        else: self.p = p
        self.beta = beta

    def v2dh(self, v):
        """outputs the height for a velocity or array of velocities"""
        logdh = self.coeff_mat[0] * np.log(v) + self.coeff_mat[1]

        dh = np.exp(logdh)
        return dh

    def dh2v(self, dh):
        """outputs the velocity for a height or set of heights"""
        logdh = np.log(dh)
        logv = (logdh - self.coeff_mat[1]) / self.coeff_mat[0]

        v = np.exp(logv)
        return v

    def model_update_rls(self, vels, dhs):
        """updates the model coefficients using the recursive
        least-squares algorithm"""
        # Algorithm from
        # https://osquant.com/papers/recursive-least-squares-linear-regression/
        vels = np.reshape(vels,-1)
        for idx, vel in enumerate(vels):
            x = np.array([[np.log(np.array(vel))], [1]])
            y = np.log(dhs[idx])
            if not np.isnan(y):
                r = 1 + (x.T @ self.p @ x) / self.beta
                k = self.p @ x / (r * self.beta)
                e = y - x.T @ self.coeff_mat
                self.coeff_mat = self.coeff_mat + k @ e
                self.p = self.p / self.beta - k @ k.T * r


def interpolate_heights(nom_height, sparse_height):
    """
    Fills in a sparse height profile with elements of the nominal height profile
    """
    for i, height in enumerate(sparse_height):
        if np.isnan(height):
            sparse_height[i] = nom_height[i]
    return sparse_height


def flame_detection_aluminum(
    raw_img, threshold=1.0e4, area_threshold=4, percentage_threshold=0.8
):
    """
    flame detection by raw counts thresholding and connected components labeling
    centroids: x,y
    bbox: x,y,w,h
    adaptively increase the threshold to 60% of the maximum pixel value
    """
    threshold = max(threshold, percentage_threshold * np.max(raw_img))
    thresholded_img = (raw_img > threshold).astype(np.uint8)

    _, labels, stats, centroids = cv2.connectedComponentsWithStats(
        thresholded_img, connectivity=4
    )

    valid_indices = np.where(stats[:, cv2.CC_STAT_AREA] > area_threshold)[0][
        1:
    ]  ###threshold connected area
    if len(valid_indices) == 0:
        return None, None

    average_pixel_values = [
        np.mean(raw_img[labels == label]) for label in valid_indices
    ]  ###sorting
    valid_index = valid_indices[
        np.argmax(average_pixel_values)
    ]  ###get the area with largest average brightness value

    # Extract the centroid and bounding box of the largest component
    centroid = centroids[valid_index]
    bbox = stats[valid_index, :-1]

    return centroid, bbox


def flame_tracking_stream(save_path, robot, robot2, positioner, flir_intrinsic, height_offset=0):
    with open(save_path + "ir_recording.pickle", "rb") as file:
        ir_recording = pickle.load(file)
    ir_ts = np.loadtxt(save_path + "ir_stamps.csv", delimiter=",")
    if ir_ts.shape[0] == 0:
        raise ValueError("No flame detected")
    joint_angle = np.loadtxt(save_path + "weld_js_exe.csv", delimiter=",")
    timeslot = [ir_ts[0] - ir_ts[0], ir_ts[-1] - ir_ts[0]]
    duration = np.mean(np.diff(timeslot))

    flame_3d = []
    job_no = []
    torch_path = []
    for start_time in timeslot[:-1]:
        start_idx = np.argmin(np.abs(ir_ts - ir_ts[0] - start_time))
        end_idx = np.argmin(np.abs(ir_ts - ir_ts[0] - start_time - duration))
    ## including for tracking purposes
    time_diffs = []
    # find all pixel regions to record from flame detection
    for i in range(start_idx, end_idx):
        start_time_tracking = time.perf_counter()
        ir_image = ir_recording[i]
        try:
            centroid, _ = flame_detection_aluminum(ir_image, percentage_threshold=0.8)
        except ValueError:
            centroid = None

        if centroid is not None:
            # find spatial vector ray from camera sensor
            vector = np.array(
                [
                    (centroid[0] - flir_intrinsic["c0"]) / flir_intrinsic["fsx"],
                    (centroid[1] - flir_intrinsic["r0"]) / flir_intrinsic["fsy"],
                    1,
                ]
            )
            vector = vector / np.linalg.norm(vector)
            # find index closest in time of joint_angle
            joint_idx = np.argmin(np.abs(ir_ts[i] - joint_angle[:, 0]))
            robot2_pose_world = robot2.fwd(joint_angle[joint_idx][9:-2], world=True)
            p2 = robot2_pose_world.p
            v2 = robot2_pose_world.R @ vector
            robot1_pose = robot.fwd(joint_angle[joint_idx][3:9])
            p1 = robot1_pose.p
            v1 = robot1_pose.R[:, 2]
            positioner_pose = positioner.fwd(joint_angle[joint_idx][-2:], world=True)

            # find intersection point
            intersection = line_intersect(p1, v1, p2, v2)
            # offset by height_offset
            intersection[2] = intersection[2]+height_offset
            intersection = positioner_pose.R.T @ (intersection - positioner_pose.p)
            torch = positioner_pose.R.T @ (robot1_pose.p - positioner_pose.p)
            time_dif_tracking=time.perf_counter()-start_time_tracking
            time_diffs.append(time_dif_tracking)
            flame_3d.append(intersection)
            torch_path.append(torch)
            job_no.append(int(joint_angle[joint_idx][1]))

    flame_3d = np.array(flame_3d)
    torch_path = np.array(torch_path)
    job_no = np.array(job_no)
    ## processing timing data
    time_diffs=np.array(time_diffs)
    # print("Average: ", np.mean(time_diffs))
    # print("Worst: ", np.max(time_diffs))
    return flame_3d, torch_path, job_no

def flame_tracking(save_path, robot, robot2, positioner, flir_intrinsic, height_offset=0):
    with open(save_path + "ir_recording.pickle", "rb") as file:
        ir_recording = pickle.load(file)
    ir_ts = np.loadtxt(save_path + "ir_stamps.csv", delimiter=",")
    if ir_ts.shape[0] == 0:
        raise ValueError("No flame detected")
    joint_angle = np.loadtxt(save_path + "weld_js_exe.csv", delimiter=",")
    timeslot = [ir_ts[0] - ir_ts[0], ir_ts[-1] - ir_ts[0]]
    duration = np.mean(np.diff(timeslot))

    flame_3d = []
    job_no = []
    torch_path = []
    for start_time in timeslot[:-1]:
        start_idx = np.argmin(np.abs(ir_ts - ir_ts[0] - start_time))
        end_idx = np.argmin(np.abs(ir_ts - ir_ts[0] - start_time - duration))
    ## including for tracking purposes
    time_diffs = []
    # find all pixel regions to record from flame detection
    for i in range(start_idx, end_idx):
        start_time_tracking = time.perf_counter()
        ir_image = ir_recording[i]
        try:
            centroid, _ = flame_detection_aluminum(ir_image, percentage_threshold=0.8)
        except ValueError:
            centroid = None

        if centroid is not None:
            # find spatial vector ray from camera sensor
            vector = np.array(
                [
                    (centroid[0] - flir_intrinsic["c0"]) / flir_intrinsic["fsx"],
                    (centroid[1] - flir_intrinsic["r0"]) / flir_intrinsic["fsy"],
                    1,
                ]
            )
            vector = vector / np.linalg.norm(vector)
            # find index closest in time of joint_angle
            joint_idx = np.argmin(np.abs(ir_ts[i] - joint_angle[:, 0]))
            robot2_pose_world = robot2.fwd(joint_angle[joint_idx][8:-2], world=True)
            p2 = robot2_pose_world.p
            v2 = robot2_pose_world.R @ vector
            robot1_pose = robot.fwd(joint_angle[joint_idx][2:8])
            p1 = robot1_pose.p
            v1 = robot1_pose.R[:, 2]
            positioner_pose = positioner.fwd(joint_angle[joint_idx][-2:], world=True)

            # find intersection point
            intersection = line_intersect(p1, v1, p2, v2)
            # offset by height_offset
            intersection[2] = intersection[2]+height_offset
            intersection = positioner_pose.R.T @ (intersection - positioner_pose.p)
            torch = positioner_pose.R.T @ (robot1_pose.p - positioner_pose.p)
            time_dif_tracking=time.perf_counter()-start_time_tracking
            time_diffs.append(time_dif_tracking)
            flame_3d.append(intersection)
            torch_path.append(torch)
            job_no.append(int(joint_angle[joint_idx][1]))

    flame_3d = np.array(flame_3d)
    torch_path = np.array(torch_path)
    job_no = np.array(job_no)
    ## processing timing data
    time_diffs=np.array(time_diffs)
    print("Average: ", np.mean(time_diffs))
    print("Worst: ", np.max(time_diffs))
    return flame_3d, torch_path, job_no


def flame_temp(save_path):
    with open(save_path + "ir_recording.pickle", "rb") as file:
        ir_recording = pickle.load(file)
    ir_ts = np.loadtxt(save_path + "ir_stamps.csv", delimiter=",")
    if ir_ts.shape[0] == 0:
        raise ValueError("No flame detected")
    joint_angle = np.loadtxt(save_path + "weld_js_exe.csv", delimiter=",")
    timeslot = [ir_ts[0] - ir_ts[0], ir_ts[-1] - ir_ts[0]]
    duration = np.mean(np.diff(timeslot))

    job_no = []
    avg_temp = []
    max_temp = []
    min_temp = []
    for start_time in timeslot[:-1]:
        start_idx = np.argmin(np.abs(ir_ts - ir_ts[0] - start_time))
        end_idx = np.argmin(np.abs(ir_ts - ir_ts[0] - start_time - duration))

    # find all pixel regions to record from flame detection
    for i in range(start_idx, end_idx):

        ir_image = ir_recording[i]
        try:
            centroid, bbox = flame_detection_aluminum(ir_image, percentage_threshold=0.8)
        except ValueError:
            centroid = None

        if centroid is not None:
            # crop to bbox image
            # plt.imshow(ir_image, cmap='inferno')
            # plt.show()
            ir_crop = ir_image[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
            # plt.imshow(ir_crop, cmap='inferno')
            # plt.show()
            joint_idx = np.argmin(np.abs(ir_ts[i] - joint_angle[:, 0]))
            job_no.append(int(joint_angle[joint_idx][1]))
            max_temp.append(np.max(ir_crop))
            min_temp.append(np.min(ir_image))
            avg_temp.append(np.average(ir_crop))
    max_temp = np.array(max_temp)
    avg_temp = np.array(avg_temp)
    min_temp = np.array(min_temp)

    job_no = np.array(job_no)
    return max_temp, avg_temp, min_temp, job_no


def calc_velocity_stream(save_path, robot):
    joint_angle = np.loadtxt(save_path + "weld_js_exe.csv", delimiter=",")

    joint_angles_prev = joint_angle[0, :]
    joint_angles_layer = joint_angle[1:, :]
    pose_prev = robot.fwd(joint_angles_prev[3:9]).p
    time_prev = joint_angles_prev[0]

    time_series_layer = []
    job_nos_layer = []
    calc_vel_layer = []
    for i in range(joint_angles_layer.shape[0]):
        # calculate instantatneous velocity
        robot1_pose = robot.fwd(joint_angles_layer[i, 3:9])
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
    calc_vel_layer = np.array(calc_vel_layer)
    job_nos_layer = np.array(job_nos_layer)
    time_series_layer = np.array(time_series_layer)
    return calc_vel_layer, job_nos_layer, time_series_layer

def calc_velocity(save_path, robot):
    joint_angle = np.loadtxt(save_path + "weld_js_exe.csv", delimiter=",")

    joint_angles_prev = joint_angle[0, :]
    joint_angles_layer = joint_angle[1:, :]
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
    calc_vel_layer = np.array(calc_vel_layer)
    job_nos_layer = np.array(job_nos_layer)
    time_series_layer = np.array(time_series_layer)
    return calc_vel_layer, job_nos_layer, time_series_layer

def line_intersect(p1,v1,p2,v2):
    #calculate the intersection of two lines, on line 1
    #find the closest point on line1 to line2
    w = p1 - p2
    a = np.dot(v1, v1)
    b = np.dot(v1, v2)
    c = np.dot(v2, v2)
    d = np.dot(v1, w)
    e = np.dot(v2, w)

    sc = (b*e - c*d) / (a*c - b*b)
    closest_point = p1 + sc * v1

    return closest_point

def vel_adjust(trans_flame, k=0.1, h_d=None):
    # assume flame rotated to flat if no h_d
    if h_d==None:
        h_d = 0
    vel_correction = k*(h_d-trans_flame)
    return vel_correction

class LiveFilter():
    '''
    Filter for live data based on a butterworth filter from Scipy.
    When calling process, the filter state is updated and the filtered measurement is output.
    3 Filters, one for each cartesian position direction
    '''
    def __init__(self):
        self.order = 2
        self.Wn = 1
        self.btype = 'lowpass'
        self.ftype = 'butter'
        self.output = 'sos'
        self.fs = 30

        self.sos = iirfilter(N=self.order,
                             Wn = self.Wn,
                             btype=self.btype,
                             ftype=self.ftype,
                             output=self.output,
                             fs=self.fs)
        self.zi_1 = np.zeros((1,2))
        self.zi_2 = np.zeros((1,2))
        self.zi_3 = np.zeros((1,2))
    def process(self,x):
        y = np.zeros(3)
        y[0], self.zi_1 = sosfilt(self.sos, [x[0]], zi=self.zi_1)
        y[1], self.zi_2 = sosfilt(self.sos, [x[1]], zi=self.zi_2)
        y[2], self.zi_3 = sosfilt(self.sos, [x[2]], zi=self.zi_3)
        return y
        
class LiveAverageFilter():
    '''
    Filters live data using averaging on a motion segment
    Counts number of samples, and keeps a cumulative sum
    averages when the output is called and clears the sum 
    '''
    def __init__(self):
        # initialize cumulative sum to zero
        self.cum_sum = np.zeros((3,1))
        # initialize sample count to zero
        self.samp_count = 0
    def log_reading(self, x):
        self.cum_sum[0]+= x[0]
        self.cum_sum[1]+= x[1]
        self.cum_sum[2]+= x[2]
        self.samp_count +=1
    def read_filter(self):
        if self.samp_count != 0:
            output = self.cum_sum/self.samp_count
            self.cum_sum = np.zeros((3,1))
            self.samp_count = 0
        else:
            output = np.zeros((3,1))
        return output
