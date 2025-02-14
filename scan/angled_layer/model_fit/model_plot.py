import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.style.use('science')
# plt.rcParams['text.usetex'] = True

def cold_model(data):
    fit_points = -0.4619*np.log(data)+1.647
    return fit_points

def hot_model(data):
    fit_points = -0.37*np.log(data)+1.215
    return fit_points

fig, ax1= plt.subplots(1,1)
fig.set_size_inches(5,3)
fig.set_dpi(300)
marker_size = 3
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
    '.',
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

# Load Data
data_hot=np.loadtxt('hot_model_data.csv', delimiter=',')
data_cold=np.loadtxt('vel_height.csv', delimiter=',')
data_cold=data_cold[:,[1,2,4,6,8,10,12,14,16]]
print(data_cold)

# Calculate Error
print(np.isnan(np.any(data_hot)))
error_hot = np.sqrt(np.mean((hot_model(data_hot[:,1])-data_hot[:,0])**2))
error_cold = np.sqrt(np.mean((cold_model(data_cold[0,:])-np.log(data_cold[1,:]))**2))
error_hot_standard = np.sqrt(np.mean((np.exp(hot_model(data_hot[:,1]))-np.exp(data_hot[:,0]))**2))
error_cold_standard = np.sqrt(np.mean((np.exp(cold_model(data_cold[0,:]))-(data_cold[1,:]))**2))

print("Hot Model Log Error: ", error_hot)
print("Cold Model Log Error: ", error_cold)
print("Hot Model Error: ", error_hot_standard)
print("Cold Model Error: ", error_cold_standard)


# Plot hot data
vel_data = np.linspace(3.5,20)
ax1.scatter(data_hot[:,1],data_hot[:,0], s=marker_size, marker=marker_styles[0], color=marker_colors[1], alpha=0.2)
ax1.plot(np.log(vel_data), hot_model(vel_data), color=marker_colors[1], label="Hot Model")
# plotting error bounds in slightly lighter opacity
ax1.fill_between(np.log(vel_data), hot_model(vel_data)+error_hot, hot_model(vel_data)-error_hot, color=marker_colors[1], alpha=0.2)
ax1.plot(np.log(vel_data), hot_model(vel_data)-error_hot, color=marker_colors[1], linestyle='dashed')
ax1.plot(np.log(vel_data), hot_model(vel_data)+error_hot, color=marker_colors[1], linestyle='dashed')

# Plot cold data
# fit_points = np.exp(-0.4619*np.log(data_cold[0,:])+1.647)
ax1.scatter(np.log(data_cold[0,:]),np.log(data_cold[1,:]), s=marker_size, marker=marker_styles[1], color=marker_colors[0], alpha=0.5)
ax1.plot(np.log(vel_data), cold_model(vel_data), color=marker_colors[0], label="Cold Model")
ax1.fill_between(np.log(vel_data), cold_model(vel_data)+error_cold, cold_model(vel_data)-error_cold, color=marker_colors[0], alpha=0.2)
ax1.plot(np.log(vel_data), cold_model(vel_data)+error_cold, color=marker_colors[0], linestyle='dashed')
ax1.plot(np.log(vel_data), cold_model(vel_data)-error_cold, color=marker_colors[0], linestyle='dashed')
# Format Plot
ax1.set_ylabel("$\ln(\Delta h)$ (mm)")
ax1.set_xlabel("$\ln(v_T)$ (mm/s)")
ax1.legend(facecolor='white', 
           framealpha=0.8,
           frameon=True,)
# ax1.legend(["Hot Measured Data", "Hot Model Fit","Cold Measured Data","Cold Model Fit"],
#            facecolor='white', 
#            framealpha=0.8,
#            frameon=True,)
ax1.grid()
ax1.set_xlim([1, 3.25])
ax1.set_ylim([-1, 2])
ax1.set_title('Model Fits')

fig.savefig('mod_plot_both_bounds.pdf')
plt.show()
