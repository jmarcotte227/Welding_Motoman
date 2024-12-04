import matplotlib.pyplot as plt
import numpy as np
import os
import scienceplots

plt.style.use('science')

dataset = "bent_tube/"
sliced_alg = "slice_ER_4043_large_hot/"
data_dir = "../../data/" + dataset + sliced_alg + "curve_sliced/"

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(projection='3d')

for file in os.listdir(data_dir):
    data = np.loadtxt(data_dir+file, delimiter=',')
    ax.plot(data[:,0], data[:,1], data[:,2],'r', linewidth='0.5')
ax.set_aspect('equal')
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')
ax.xaxis.set_major_locator(plt.MaxNLocator(3))
ax.yaxis.set_major_locator(plt.MaxNLocator(2))
ax.zaxis.set_major_locator(plt.MaxNLocator(5))
plt.savefig('pathplan.pdf', pad_inches=0.5)
