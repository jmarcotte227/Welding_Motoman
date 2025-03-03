import matplotlib.pyplot as plt
import numpy as np
import os
import scienceplots

plt.style.use('science')

dataset = "s_curve_angled/"
sliced_alg = "slice/"
data_dir = "../../data/" + dataset + sliced_alg + "curve_sliced/"

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(projection='3d')
dir_flag = False

for file in os.listdir(data_dir):
    data = np.loadtxt(data_dir+file, delimiter=',')
    if dir_flag:
        ax.plot(data[1:,0], data[1:,1], data[1:,2],'r', linewidth='0.5')
        ax.scatter(data[0,0], data[0,1], data[0,2], color='b', s=5)
    else:
        ax.plot(data[:-1,0], data[:-1,1], data[:-1,2],'r', linewidth='0.5')
        ax.scatter(data[-1,0], data[-1,1], data[-1,2], color='b', s=5)
    dir_flag = not dir_flag
    print(dir_flag)
ax.set_aspect('equal')
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')
ax.xaxis.set_major_locator(plt.MaxNLocator(3))
ax.yaxis.set_major_locator(plt.MaxNLocator(2))
ax.zaxis.set_major_locator(plt.MaxNLocator(5))
plt.savefig('pathplan_marked_s.eps', pad_inches=0.5)
plt.show()
