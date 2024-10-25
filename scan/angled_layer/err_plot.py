import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True

fig, ax= plt.subplots(1,1)
fig.set_size_inches(10,6)
fig.set_dpi(200)
plt_params = [
    'b--',
    'r',
    'g-.'
]

err_set = [
    'error_data/ER4043_bent_tube_2024_08_28_12_24_30_err.csv',
    'error_data/ER4043_bent_tube_2024_09_04_12_23_40_err.csv',
    'error_data/ER4043_bent_tube_hot_2024_10_21_13_25_58_err.csv',
]
for idx,err in enumerate(err_set):
    err_data=np.loadtxt(err, delimiter=',')
    ax.plot(np.linspace(1,80,80),err_data, plt_params[idx])
ax.set_xlabel("Layer Number")
ax.set_ylabel("RMSE (mm)")
ax.legend(["Open-Loop Cold Model", "Closed-Loop Cold Model", "Closed-Loop Hot Model"])
ax.grid()
ax.set_title('Layer Error Comparison')
fig.savefig('rms_plot.png')
plt.show()
