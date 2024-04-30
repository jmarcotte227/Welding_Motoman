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

width_data_file = open('4_speed_wall_width.pickle', 'rb')
width_data = pickle.loads(width_data_file.read())
data_lim = [-50,30]
widths = []
widths_range = []
xcoords_range = []
widths_avg = []
x_coord = []

plot_flag = True

##Lump data for all layer widths
for key in width_data:
    x_coord.extend(list(width_data[key].keys()))
    widths.extend(list(width_data[key].values()))

for idx, val in enumerate(x_coord):
    if data_lim[0] < val < data_lim[1]:
        xcoords_range.append(val)
        widths_range.append(widths[idx])

widths = np.array(widths_range)
x_coord = np.array([x*-1 for x in xcoords_range])



fig,ax = plt.subplots(1,1)
# Inverse Exponential
popt, pcov = curve_fit(lin, x_coord, widths)
x_data = np.linspace(-data_lim[1], -data_lim[0])

ax.plot(x_data, lin(x_data, *popt), 'r')
print(*popt)
ax.scatter(x_coord,widths)
plt.title("All Layer Width Data | Linear")
plt.xlabel("x position (mm/s)")
plt.ylabel("Bead Width (mm)")
#plt.gca().set_aspect('equal')
plt.gca().set_ylim([5,10])
plt.grid()
plt.show()

slopes = []
intercepts = [] 
averages = []
coord_avg = []
#Individual layer analysis
fig,ax = plt.subplots(1,1)
for key in width_data:
    x_coord = list(width_data[key].keys())
    widths = list(width_data[key].values())
    popt, pcov = curve_fit(lin, x_coord, widths)

    slopes.append(popt[0])
    intercepts.append(popt[1])

    if plot_flag:
        
        x_data = np.linspace (20, 90)
        ax.plot(x_data, lin(x_data, *popt), label = str(key))
        #ax.scatter(x_coord,widths)
        plt.title("All Layer Width Data | Linear")
        plt.xlabel("x position (mm/s)")
        plt.ylabel("Bead Width (mm)")
plt.legend()
plt.show()
# print(coord_avg)
# fig,ax = plt.subplots(1,1)
# ax.plot(coord_avg,averages)
# plt.show()
# fig,ax = plt.subplots(1,1)
# ax.plot(slopes)
# plt.show()
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


#average slope plot()