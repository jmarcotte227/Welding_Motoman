"""
Contains the relevant python functions and objects for welding parts
using non-uniform height profiles.
"""

import numpy as np
import cv2


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

    def __init__(self, lam=0.05, beta=1, a=-0.4619, b=1.647):
        # Beta == 1 for non-exponentail updates
        self.coeff_mat = np.array([a, b])
        self.nom_a = a
        self.nom_b = b
        self.lam = lam
        self.p = np.diag(np.ones(self.coeff_mat.shape[0]) * self.lam)
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
        return None, None, None, None

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
