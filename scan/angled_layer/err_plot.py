import numpy as np
import matplotlib.pyplot as plt

# plt.rcParams['text.usetex'] = True

fig, ax= plt.subplots(1,1)
fig.set_size_inches(10,6)
fig.set_dpi(200)
plt_params = [
    'b--',
    'r',
    'g-.'
]

err_set = [
    # 'error_data/ER4043_bent_tube_2024_08_28_12_24_30_err.csv',
    'error_data/ER4043_bent_tube_large_cold_2024_11_07_10_21_39_err.csv',
    'error_data/ER4043_bent_tube_large_hot_2024_11_06_12_27_19_err.csv',
]
for idx,err in enumerate(err_set):
    err_data=np.loadtxt(err, delimiter=',')
    ax.plot(np.linspace(1,106,106),err_data, plt_params[idx])
ax.set_xlabel("Layer Number")
ax.set_ylabel("RMSE (mm)")
ax.legend(["Closed-Loop Cold Model", "Closed-Loop Hot Model"])
ax.grid()
ax.set_title('Layer Error Comparison')
fig.savefig('rms_plot.png')
plt.show()
