'''
Plots error in previous layer and returns the planned velocity profile for the next layer
'''

import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.style.use('science')
# plt.rcParams['text.usetex'] = True

layer = 100

fig, ax1= plt.subplots(1,1)

fig.set_size_inches(5,2)
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
marker_styles = [
    'o',
    '^',
    's',
    'D',
    'v',
    '2',
    'p'

]
marker_colors = [
    '#0C5DA5',
    '#00B945',
    '#FF9500',
    '#FF2C00',
    '#845B97',
    '#474747',
    '#9e9e9e'

]
marker_size = 10
data_sets = [
    # 'ER4043_bent_tube_large_cold_OL_2024_11_14_11_56_43',
    # 'ER4043_bent_tube_large_hot_OL_2024_11_14_13_05_38',
    # 'ER4043_bent_tube_large_cold_2024_11_07_10_21_39',
    'ER4043_bent_tube_large_hot_2024_11_06_12_27_19'
]
# plot blank plots to increment color cycler
ax1._get_lines.get_next_color()
ax1._get_lines.get_next_color()
ax1._get_lines.get_next_color()


plan_data = np.loadtxt('../../../data/bent_tube/slice_ER_4043_large_hot/curve_sliced/slice2_0.csv', delimiter=',')
height_data = plan_data[1:-1, 2]-4
for idx, test in enumerate(data_sets):
    err_data=np.loadtxt(f'../error_data/{test}_layer_err.csv', delimiter=',')
    layer_err = err_data[layer-2,:]
    ax1.scatter(
        np.linspace(2,49,48),
        height_data-layer_err,
        marker=marker_styles[3+idx],
        color=marker_colors[3+idx],
        s=marker_size
    )
# vel_data_cold = np.loadtxt(f'../../../../recorded_data/{data_sets[0]}/layer_1/velocity_profile.csv', delimiter=',')
vel_data_hot = np.loadtxt(f'../../../../recorded_data/{data_sets[0]}/layer_1/velocity_profile.csv', delimiter=',')

ax1.set_xlabel("Segment Number")
ax1.set_ylabel("$\Delta h_{d}$ (mm)")
# ax1.set_ylabel("Error (mm)")
# fig.legend(["CC", "CH", "Nominal Height Profile", "Cold Model Nominal Velocity", "Hot Model Nominal Velocity"],
#            facecolor='white',
#            framealpha=0.8,
#            frameon=True,
#            loc='lower center',
#            ncol=2,
#            bbox_to_anchor=(0.5,-0.2)
#            )
ax1.grid()
ax1.set_title('Target Deposition Height')
plt.tight_layout()
fig.savefig(f'height_plot_layer.pdf')
plt.show()
