import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.style.use('science')
# plt.rcParams['text.usetex'] = True

fig, (ax1, ax2)= plt.subplots(2,1, sharex=True)
fig.set_size_inches(5,4)
fig.set_dpi(300)
marker_size = 2
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

err_set = [
    'error_data/ER4043_bent_tube_large_cold_OL_2024_11_14_11_56_43_err.csv',
    'error_data/ER4043_bent_tube_large_hot_OL_2024_11_14_13_05_38_err.csv',
    'error_data/ER4043_bent_tube_large_cold_2024_11_07_10_21_39_err.csv',
    'error_data/ER4043_bent_tube_large_hot_2024_11_06_12_27_19_err.csv',
    'error_data/ER4043_bent_tube_2024_09_04_12_23_40_err.csv',
]
for idx,err in enumerate(err_set[:-1]):
    err_data=np.loadtxt(err, delimiter=',')
    ax1.scatter(np.linspace(1,106,106),err_data, s=marker_size, marker=marker_styles[idx])
    ax2.scatter(np.linspace(1,106,106),err_data, s=marker_size, marker=marker_styles[idx])
# err_data=np.loadtxt(err_set[-1], delimiter=',')
# ax1.plot(np.linspace(1,80,80), err_data)
# ax2.plot(np.linspace(1,80,80), err_data)

ax2.set_xlabel("Layer Number")
ax1.set_ylabel("RMSE (mm)")
ax2.set_ylabel("RMSE (mm)")
ax1.legend(["OC", "OH","CC", "CH"], 
           facecolor='white', 
           framealpha=0.8,
           frameon=True,
           # loc='lower center',
           # ncol=2,
           # bbox_to_anchor=(0.5,-0.8)
           )
ax1.grid()
ax2.grid()
ax1.set_title('Layer Error')
ax2.set_title('Layer Error Zoomed')

ax2.set_ylim(0,2)
fig.savefig('rms_plot_mod.pdf')
plt.show()
