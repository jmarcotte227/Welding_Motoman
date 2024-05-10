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

width_data_file = open('2_speed_wall_width.pickle', 'rb')
width_data = pickle.loads(width_data_file.read())
data_lim = [-10,90]
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

# find average widths for both sections
width_low_sum = 0
width_low_count = 0
width_high_sum = 0
width_high_count = 0

lower_bound = -data_lim[1] + 10
print(lower_bound)
upper_bound = -data_lim[0] - 10
print(upper_bound)
cutoff = -data_lim[0] - 50
print(cutoff)

print(max(x_coord))
print(min(x_coord))

for idx, coord in enumerate(x_coord):
    if lower_bound < coord < cutoff:
        width_low_sum += widths[idx]
        width_low_count +=1

    if cutoff < coord < upper_bound:
        width_high_sum += widths[idx]
        width_high_count += 1

average_low = width_low_sum/width_low_count
average_high = width_high_sum/width_high_count

print("Average Low: ", average_low)
print("Average High: ", average_high)

plot_offset = 90

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig,ax = plt.subplots(1,1)
fig.dpi=300
# Inverse Exponential
popt, pcov = curve_fit(lin, x_coord, widths)
x_data = np.linspace(-data_lim[1], -data_lim[0])

#ax.plot(x_data, lin(x_data, *popt), 'r')

ax.scatter(x_coord+plot_offset,widths)
ax.plot([lower_bound+plot_offset, cutoff+plot_offset], [average_low, average_low], 'r')
ax.plot([cutoff+plot_offset, upper_bound+plot_offset], [average_high, average_high], 'r')
ax.legend(["Width Data", "Average Width"])
# plt.title("All Layer Width Data | Linear")
plt.xlabel("X Position (mm)")
plt.ylabel("Bead Width (mm)")
#plt.gca().set_aspect('equal')
plt.gca().set_ylim([4,7.5])
#plt.grid()
plt.show()

# slopes = []
# intercepts = [] 
# averages = []
# coord_avg = []
# #Individual layer analysis
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