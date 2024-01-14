import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.optimize import curve_fit

def exp(x, a, b, c):
    return a*np.exp(-b*x) + c

def lin(x, a, b):
    return a*x + b

def quad(x, a, b, c):
    return a*x**2+b*x+c

width_data_file = open('all_layer_width.pickle', 'rb')
width_data = pickle.loads(width_data_file.read())

widths = []
widths_avg = []
x_coord = []

for key in width_data:

    x_coord.extend(list(width_data[key].keys()))
    widths.extend(list(width_data[key].values()))
widths = np.array(widths)
x_coord = np.array(x_coord)


fig,ax = plt.subplots(1,1)
# Inverse Exponential
popt, pcov = curve_fit(lin, x_coord, widths)
x_data = np.linspace (20, 90)
ax.plot(x_data, lin(x_data, *popt), 'r')
ax.scatter(x_coord,widths,)
plt.title("Speed vs Bead Width | Cropped | Quadratic")
plt.xlabel("Torch Speed (mm/s)")
plt.ylabel("Bead Width (mm/s)")
plt.show()

# Log-Log Linear
# speeds_log = np.log(speeds)
# widths_log = np.log(widths)
# popt, pcov = curve_fit(lin, speeds_log, widths_log)
# plt.plot(speeds_log, lin(speeds_log, *popt))
# plt.scatter(speeds_log,widths_log)
# plt.title("Speed vs Bead Width | Every Layer | Log-Log linear fit")
# plt.xlabel("log Torch Speed")
# plt.ylabel("log Bead Width")
# plt.show()


print(popt)