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

err_set = [
    'error_data/ER4043_bent_tube_2024_09_04_12_23_40_err.csv',
]
for idx,err in enumerate(err_set):
    err_data=np.loadtxt(err, delimiter=',')
    ax.plot(np.linspace(1,80,80),err_data, linestyle=plt_styles[idx])


ax.set_xlabel("Layer Number")
ax.set_ylabel("RMSE (mm)")
ax.grid()
ax.set_title('Layer Error')
fig.savefig('rms_plot_cold_90.pdf')
plt.show()
