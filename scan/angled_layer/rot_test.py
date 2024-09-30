import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('../../toolbox')
from angled_layers import avg_by_line, rotate

rot_point = np.array([3,0,10])
angle = np.deg2rad(20)
data = np.array([0, 10,5])

rotated_data = rotate(rot_point[[0,2]],data[[0,2]], angle)


fig, ax = plt.subplots(1,1)
ax.plot(data[0], data[2], 'ro')
ax.plot(rot_point[0], rot_point[2], 'bo')
ax.plot(rotated_data[0], rotated_data[1], 'go')
ax.set_aspect('equal')
plt.show()


