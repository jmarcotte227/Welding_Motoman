import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
def exp(x, a, b, c):
    return a*np.exp(-b*x) + c

def lin(x, a, b):
    return a*x + b

def quad(x, a, b, c):
    return a*x**2+b*x+c

width_data = {
    #ER 4043 | 160 ipm
    #torch speed : [speed, maximum, highest 3 average] 
    "L1" : (4, 4.545558091463416, 4.537514957153295),  #layer 1
    "L2" : (4, 6.065867812641138, 6.06353509775203), #layer 2
    "L3" : (6, 5.200371404430731, 5.102569354890101),
    "L4" : (6, 6.141707415099384, 6.135023037880518),
    "L5" : (8, 4.844807705110973, 4.799933405216531),
    "L6" : (8, 4.84760301122418, 4.8436681619256),
    "L7" : (10, 3.916360355315389, 3.9034265364098686),
    "L8" : (10, 4.14744694872166, 4.147045980025386),
    "L9" : (12, 3.8827902273058927, 3.8612650170680496),
    "L10": (12, 4.014599346504976, 4.012936919910595),
    "L11": (14, 3.75910614376284, 3.732025546526058),
    "L12": (14, 3.8154558335087376, 3.8072503276251966),
    "L13": (16, 3.590848620371693, 3.570709135529396),
    "L14": (16, 3.834890205005155, 3.8218012113415814),
    "L15": (18, 3.6327612925814545, 3.614910287188151),
    "L16": (18, 3.391501806415093, 3.368572935385958),
    "L17": (20, 3.341520973414749, 3.317045816225828),
}
width_data_second = {
    "L2" : (4, 6.065867812641138, 6.06353509775203),
    "L4" : (6, 6.141707415099384, 6.135023037880518),
    "L6" : (8, 4.84760301122418, 4.8436681619256),
    "L8" : (10, 4.14744694872166, 4.147045980025386),
    "L10": (12, 4.014599346504976, 4.012936919910595),
    "L12": (14, 3.8154558335087376, 3.8072503276251966),
    "L14": (16, 3.834890205005155, 3.8218012113415814),
    "L16": (18, 3.391501806415093, 3.368572935385958),

}

width_data_cropped = {
    "L7" : (10, 3.916360355315389, 3.9034265364098686),
    "L8" : (10, 4.14744694872166, 4.147045980025386),
    "L9" : (12, 3.8827902273058927, 3.8612650170680496),
    "L10": (12, 4.014599346504976, 4.012936919910595),
    "L11": (14, 3.75910614376284, 3.732025546526058),
    "L12": (14, 3.8154558335087376, 3.8072503276251966),
    "L13": (16, 3.590848620371693, 3.570709135529396),
    "L14": (16, 3.834890205005155, 3.8218012113415814),
    "L15": (18, 3.6327612925814545, 3.614910287188151),
    "L16": (18, 3.391501806415093, 3.368572935385958),
    "L17": (20, 3.341520973414749, 3.317045816225828),
}

widths = []
widths_avg = []
speeds = []

for key in width_data_cropped:
    widths.append(width_data[key][1])
    speeds.append(width_data[key][0])
widths = np.array(widths)
speeds = np.array(speeds)

# Inverse Exponential
popt, pcov = curve_fit(quad, speeds, widths)
x_data = np.linspace (0, 20)
plt.plot(x_data, quad(x_data, *popt))
plt.scatter(speeds,widths)
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