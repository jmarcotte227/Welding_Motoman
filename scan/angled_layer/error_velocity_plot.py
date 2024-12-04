'''
Plots error in previous layer and returns the planned velocity profile for the next layer
'''

import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.style.use('science')
# plt.rcParams['text.usetex'] = True

layer = 100

fig, (ax1, ax2)= plt.subplots(2,1, sharex=True)

fig.set_size_inches(5,4)
fig.set_dpi(300)
plt_colors = [
    'blue',
    'red',
    'green',
    'm'
]
plt_styles = [
    'solid',
    'dotted',
    'dashed',
    'dashdot'
]
data_sets = [
    # 'ER4043_bent_tube_large_cold_OL_2024_11_14_11_56_43',
    # 'ER4043_bent_tube_large_hot_OL_2024_11_14_13_05_38',
    'ER4043_bent_tube_large_cold_2024_11_07_10_21_39',
    'ER4043_bent_tube_large_hot_2024_11_06_12_27_19'
]
# plot blank plots to increment color cycler
ax1._get_lines.get_next_color()
ax1._get_lines.get_next_color()
ax2._get_lines.get_next_color()
ax2._get_lines.get_next_color()


plan_data = np.loadtxt('../../data/bent_tube/slice_ER_4043_large_hot/curve_sliced/slice2_0.csv', delimiter=',')
height_data = plan_data[1:-1, 2]-4
for idx, test in enumerate(data_sets):
    err_data=np.loadtxt(f'error_data/{test}_layer_err.csv', delimiter=',')
    layer_err = err_data[layer-2,:]
    ax1.plot(np.linspace(2,49,48),height_data-layer_err, linestyle=plt_styles[2+idx])
    vel_data = np.loadtxt(f'../../../recorded_data/{test}/layer_{layer}/velocity_profile.csv', delimiter=',',)
    ax2.plot(np.linspace(2,49,48),np.flip(vel_data[1:-1]), linestyle=plt_styles[2+idx], label='_nolegend_')
# increment artist on plot 2 once
ax2._get_lines.get_next_color()
# get default velocity profiles for cold and hot models
vel_data_cold = np.loadtxt(f'../../../recorded_data/{data_sets[0]}/layer_1/velocity_profile.csv', delimiter=',')
vel_data_hot = np.loadtxt(f'../../../recorded_data/{data_sets[1]}/layer_1/velocity_profile.csv', delimiter=',')
ax2.plot(np.linspace(2,49,48),np.flip(vel_data_cold[1:-1]), linestyle=plt_styles[0],)
ax2.plot(np.linspace(2,49,48),np.flip(vel_data_hot[1:-1]), linestyle=plt_styles[1],)

ax1.plot(np.linspace(2,49,48), height_data)
ax2.set_xlabel("Segment Number")
ax1.set_ylabel("$\Delta h_{d,100}$ (mm)")
# ax1.set_ylabel("Error (mm)")
ax2.set_ylabel("Torch Speed (mm/s)")
fig.legend(["Closed-Loop Cold Model", "Closed-Loop Hot Model", "Nominal Height Profile", "Cold Model Nominal Velocity", "Hot Model Nominal Velocity"],
           facecolor='white',
           framealpha=0.8,
           frameon=True,
           loc='lower center',
           ncol=2,
           bbox_to_anchor=(0.5,-0.2)
           )
ax1.grid()
ax2.grid()
ax1.set_title('Layer Error')
ax2.set_title('Planned Velocity Profile')
plt.tight_layout()
fig.savefig(f'vel_plot_layer_{layer}_dh.pdf')
plt.show()
