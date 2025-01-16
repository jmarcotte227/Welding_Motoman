'''
Plots recursive least squares coefficients
'''

import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.style.use('science')
# plt.rcParams['text.usetex'] = True

fig, (ax1, ax2)= plt.subplots(2,1, sharex=True)

fig.set_size_inches(5,4)
fig.set_dpi(300)
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

marker_size = 5

layer_start = 1
layer_end=80
coeffs = []
for i in range(layer_start, layer_end):
    layer_coeffs=np.loadtxt(f'../../../recorded_data/ER4043_bent_tube_2024_08_22_11_12_27/layer_{i}/coeff_mat.csv')
    # layer_coeffs=np.loadtxt(f'../../../recorded_data/ER4043_bent_tube_2024_09_03_13_26_16/layer_{i}/coeff_mat.csv')

    coeffs.append(layer_coeffs)

coeffs = np.array(coeffs)
avg_a = np.average(coeffs[40:60,0])
avg_b = np.average(coeffs[40:60,1])
print(avg_a)
print(avg_b)
ax1.scatter(
    np.linspace(1,79,79),
    coeffs[:,0],
    marker=marker_styles[0],
    color=marker_colors[0],
    s=marker_size+5
)
ax1.plot(
    [1,79],
    [-0.4619, -0.4619]
)
ax1.plot(
    [1,79],
    [avg_a, avg_a]
)
ax2.scatter(
    np.linspace(1,79,79),
    coeffs[:,1],
    marker=marker_styles[1],
    color=marker_colors[1],
    s=marker_size+5
)

ax2.plot(
    [1,79],
    [1.647, 1.647]
)
ax2.plot(
    [1,79],
    [avg_b, avg_b]
)

ax2.set_xlabel("Layer Number")
ax1.set_ylabel("$a$")
ax2.set_ylabel("$b$")
ax1.legend(["RLS $a$", "Nominal $a$", "Steady-State $a$"], 
           facecolor='white', 
           framealpha=0.8,
           frameon=True,
           # loc='lower center',
           # ncol=2,
           # bbox_to_anchor=(0.5,-0.8)
           )
ax2.legend(["RLS $b$", "Nominal $b$", "Steady-State $b$"], 
           facecolor='white',
           framealpha=0.8,
           frameon=True,
           # loc='lower center',
           # ncol=2,
           # bbox_to_anchor=(0.5,-0.8)
           )
ax1.grid()
ax2.grid()

plt.tight_layout()
fig.savefig(f'model_coeff_evolution.png')
plt.show()
