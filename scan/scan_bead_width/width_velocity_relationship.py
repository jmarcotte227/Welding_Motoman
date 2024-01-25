import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.optimize import curve_fit
import sys
sys.path.append('../../weld')
import weld_dh2v
import weld_w2v

def exp(x, a, b, c):
    return a**(b*x) + c

def lin(x, a, b):
    return a*x + b

def quad(x, a, b, c):
    return a*x**2+b*x+c

width_data_file = open('all_layer_width_high_res.pickle', 'rb')
velocity_profile_file = open('layer_velocity_profile.pickle', 'rb')
width_data = pickle.loads(width_data_file.read())
velocity_profile = pickle.loads(velocity_profile_file.read())

velocity_map = np.zeros((len(velocity_profile),2))
position = 0
point_distance = 0.5
for idx, velocity in enumerate(velocity_profile):
    velocity_map[idx,0] = position
    velocity_map[idx,1] = velocity
    position+=point_distance
dH_min = weld_dh2v.v2dh_loglog(velocity_map[0,1], 160, "ER_4043")
dH_max = weld_dh2v.v2dh_loglog(velocity_map[-1,1], 160, "ER_4043")
h_profile = np.linspace(dH_min, dH_max, 1000)
vel_profile = weld_dh2v.dh2v_loglog(h_profile, 160, "ER_4043")

vel_interpolate = {}
position = 0
point_distance = 0.1
for vel in vel_profile:
    vel_interpolate[round(position,1)] = vel
    position+=point_distance

fig,ax = plt.subplots(1,1)

ax.scatter(velocity_map[:,0], velocity_map[:,1])
ax.plot(vel_interpolate.keys(), vel_interpolate.values())
plt.show()


widths = []
widths_avg = []
x_coord = []
velocity = []

plot_flag = True

##Lump data for all layer widths
for key in width_data:
    x_coord.extend(list(width_data[key].keys()))
    for x in width_data[key].keys():
        x_round = round(x,1)
        velocity.append(vel_interpolate[x_round])
        
    widths.extend(list(width_data[key].values()))
widths = np.array(widths)
x_coord = np.array(x_coord)

velocity = np.array(velocity)

fig,ax = plt.subplots(1,2)

# linear fit of lump data
# popt, pcov = curve_fit(lin, x_coord, widths)
# x_data = np.linspace (20, 90)
# ax.plot(x_data, lin(x_data, *popt), 'r')
# print(*popt)
ax[0].scatter(x_coord,widths)
ax[1].scatter(velocity,widths)
plt.title("All Layer Width Data | Linear")
plt.xlabel("x position (mm/s)")
plt.ylabel("Bead Width (mm)")
plt.show()

slopes = []
intercepts = [] 
#Individual layer analysis
# fig,ax = plt.subplots(1,1)
# for key in width_data:
#     x_coord = list(width_data[key].keys())
#     widths = list(width_data[key].values())
#     popt, pcov = curve_fit(lin, x_coord, widths)

#     slopes.append(popt[0])
#     intercepts.append(popt[1])

#     if plot_flag:
        
#         x_data = np.linspace (20, 90)
#         ax.plot(x_data, lin(x_data, *popt), label = str(key))
#         #ax.scatter(x_coord,widths)
#         plt.title("All Layer Width Data | Linear")
#         plt.xlabel("x position (mm/s)")
#         plt.ylabel("Bead Width (mm)")
# plt.legend()
# plt.show()

fig,ax = plt.subplots(1,3)
# Log-Log Linear
speeds_log = np.log(velocity)
widths_log = np.log(widths)
popt, pcov = curve_fit(lin, speeds_log, widths_log)
ax[0].plot(speeds_log, lin(speeds_log, *popt), 'r')
ax[0].scatter(speeds_log,widths_log)
ax[0].set_xlabel("log Torch Speed")
ax[0].set_ylabel("log Bead Width")

# Log-Log Linear Error
ll_mse = np.linalg.norm((widths_log-lin(speeds_log, *popt))**2)
ll_rmse = np.linalg.norm(widths_log-lin(speeds_log, *popt))/np.sqrt(len(widths_log))
print("Log-Log MSE: ", ll_mse)
print("Log-Log RMSE: ", ll_rmse)
print("popt: ", popt)

#log-linear linear
popt, pcov = curve_fit(lin, speeds_log, widths)
ax[1].plot(speeds_log, lin(speeds_log, *popt), 'r')
ax[1].scatter(speeds_log,widths)
ax[1].set_xlabel("log Torch Speed")
ax[1].set_ylabel("Bead Width")

# Log-linear Linear Error
ll_mse = np.linalg.norm((widths-lin(speeds_log, *popt))**2)
ll_rmse = np.linalg.norm(widths-lin(speeds_log, *popt))/np.sqrt(len(widths_log))
print("Log-linear MSE: ", ll_mse)
print("Log-linear RMSE: ", ll_rmse)


#linear-linear
popt, pcov = curve_fit(lin, velocity, widths)
ax[2].plot(velocity, lin(velocity, *popt), 'r')
ax[2].scatter(velocity,widths)
ax[2].set_xlabel("Torch Speed")
ax[2].set_ylabel("Bead Width")

# Log-linear Linear Error
ll_mse = np.linalg.norm((widths-lin(velocity, *popt))**2)
ll_rmse = np.linalg.norm(widths-lin(velocity, *popt))/np.sqrt(len(widths_log))
print("Linear-linear MSE: ", ll_mse)
print("Linear-linear RMSE: ", ll_rmse)
plt.show()
plt.close()
############################################Verification


fig_ver, ax_ver = plt.subplots(1,1)

ax_ver.scatter(velocity,widths)
vel_ver = np.linspace(5, 15)
w_ver = weld_w2v.v2w_loglog(vel_ver)
ax_ver.plot(vel_ver, w_ver, 'r')
plt.show()
plt.close()