import matplotlib.pyplot as plt
import numpy as np
layers = [10, 20, 30,40, 50]
fig,ax=plt.subplots(1,1)
for layer in layers:
    vel_profile = np.loadtxt(
        f"../../../recorded_data/ER4043_bent_tube_small_2024_09_12_12_14_40/layer_{layer}/velocity_profile.csv",
        delimiter = ','
    )
    ax.plot(vel_profile)
ax.set_xlabel("Segment Number")
ax.set_ylabel("Velocity (mm/s)")
ax.set_title(f"Layers {layers}")
ax.legend(layers)
plt.show()
