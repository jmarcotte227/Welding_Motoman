import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import scienceplots

plt.style.use('science')
# colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# print('\n'.join(color for color in colors))
# exit()
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 15})

fig, ax1 = plt.subplots(1,1)
fig.set_size_inches(5,4)
fig.set_dpi(300)
marker_size = 2
plt_colors = [
    '#0C5DA5',
    '#00B945',
    '#FF9500',
    '#FF2C00',
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
    'D'
]
# labels = [
#         "Open-Loop Cold Model",
#         "Open-Loop Hot Model",
#         "Closed-Loop Cold Model",
#         "Closed-Loop Hot Model"
#         ]
labels = [
        'Open Loop',
        'Closed Loop',
        ]

err_set = [
    'error_data/ER4043_bent_tube_large_cold_OL_2024_11_14_11_56_43_err.csv',
    # 'error_data/ER4043_bent_tube_large_hot_OL_2024_11_14_13_05_38_err.csv',
    # 'error_data/ER4043_bent_tube_large_cold_2024_11_07_10_21_39_err.csv',
    'error_data/ER4043_bent_tube_large_hot_2024_11_06_12_27_19_err.csv',
    # 'error_data/ER4043_bent_tube_2024_09_04_12_23_40_err.csv',
]
for idx,err in enumerate(err_set):
    err_data=np.loadtxt(err, delimiter=',')
    ax1.scatter(
            np.linspace(1,106,106),
            err_data, 
            s=marker_size, 
            marker=marker_styles[idx],
            label = labels[idx],
            color = plt_colors[idx]
            )
    ax1.plot(
            np.linspace(1,106,106),
            err_data,
            alpha=0.3,
            color = plt_colors[idx]
            )
# err_data=np.loadtxt(err_set[-1], delimiter=',')
# ax1.plot(np.linspace(1,80,80), err_data)
# ax2.plot(np.linspace(1,80,80), err_data)

ax1.set_ylabel("RMSE (mm)")
ax1.legend(facecolor='white',
           framealpha=0.8,
           frameon=True,
           # loc='lower center',
           # ncol=2,
           # bbox_to_anchor=(0.5,-0.8)
           )
ax1.grid()
ax1.set_title('Layer Error')
ax1.set_xlabel("Layer Number")
# ax2.set_title('Layer Error Zoomed')
fig.savefig('rms_plot_simple.png', dpi=fig.dpi)
plt.show()
