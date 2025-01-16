import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.style.use('science')
# plt.rcParams['text.usetex'] = True

fig, ax1= plt.subplots(1,1)
fig.set_size_inches(5,3)
fig.set_dpi(300)
marker_size = 10
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
data=np.loadtxt('hot_model_data.csv', delimiter=',')
#ax1.scatter(,err_data, s=marker_size, marker=marker_styles[idx])
ax1.scatter(data[:,1],data[:,0], s=marker_size, marker=marker_styles[0])

ax1.plot([0.5,3], [-0.37*0.5+1.215, -0.37*3+1.215], color=marker_colors[1])
ax1.set_xlim([1,3])
# err_data=np.loadtxt(err_set[-1], delimiter=',')
# ax1.plot(np.linspace(1,80,80), err_data)
# ax2.plot(np.linspace(1,80,80), err_data)
ax1.set_ylabel("$\ln (\Delta h) (\ln$ (mm))")
ax1.set_xlabel("$\ln (v_T) (\ln$ (mm/s))")
ax1.legend(["Measured Data", "Model Fit"],
           facecolor='white', 
           framealpha=0.8,
           frameon=True,)
ax1.grid()
ax1.set_title('Hot Model Fit')

fig.savefig('mod_plot_hot.pdf')
plt.show()
