"""
Contains the relevant python functions and objects for welding parts
using non-uniform height profiles.
"""

import numpy as np


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


def avg_by_line(labels, data_array, bins):
    """
    Averages the data in the variable "data" into the bins in "bins"
    according to the values in "job_line"
    """
    data_dim = data_array.shape[1]

    output = np.zeros((bins.shape[0], data_dim))
    for idx,_bin in enumerate(bins):
        avg_idxs = np.where(labels==_bin)[0]
        if avg_idxs.shape[0] == 0:
            output[idx,:] = [None]*data_dim
        else:
            total = np.sum(data_array[avg_idxs,:], axis=0)
            output[idx,:] = total/avg_idxs.shape[0]
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
