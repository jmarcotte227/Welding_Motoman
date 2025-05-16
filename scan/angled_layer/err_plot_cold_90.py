import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.style.use('science')
# plt.rcParams['text.usetex'] = True

fig, ax = plt.subplots(1,1, sharex=True)
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
    'D'
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
marker_size = 5
err_set = [
    'error_data/ER4043_bent_tube_2024_09_04_12_23_40_err.csv',
]
for idx,err in enumerate(err_set):
    err_data=np.loadtxt(err, delimiter=',')
    ax.scatter(np.linspace(1,80,80),err_data, marker=marker_styles[idx+2], color=marker_colors[idx+1], s=marker_size)

ax.legend(["Closed-Loop Cold Model 90$^{\circ}$"], 
           facecolor='white', 
           framealpha=0.8,
           frameon=True,
           # loc='lower center',
           # ncol=2,
           # bbox_to_anchor=(0.5,-0.8)
           )

ax.set_xlabel("Layer Number")
ax.set_ylabel("RMSE (mm)")
ax.grid()
ax.set_title('Layer Error')
fig.savefig('rms_plot_cold_90.pdf')
plt.show()
