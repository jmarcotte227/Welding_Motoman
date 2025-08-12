import numpy as np
import matplotlib.pyplot as plt
import scienceplots


plt.style.use('science')
# plt.rcParams['text.usetex'] = True

fig, ax1= plt.subplots(1,1)
fig.set_size_inches(3,1.5)
fig.set_dpi(300)

vels=np.linspace(5,15,100)
a_cold=-0.4619
b_cold=1.647
a_hot=-0.3700
b_hot=1.215
f_cold=np.exp(a_cold*np.log(vels)+b_cold)
f_hot=np.exp(a_hot*np.log(vels)+b_hot)

ax1.plot(vels, f_cold)
ax1.plot(vels, f_hot)
ax1.grid()
# ax1.legend(["$\\bar{f}_{cold}$","$\\bar{f}_{hot}$"])
ax1.legend(["$\\bar{f}_{cold}$","$\\bar{f}_{hot}$"],
           facecolor='white', 
           framealpha=0.8,
           frameon=True,)
ax1.set_xlabel("$v_t$ (mm/s)")
ax1.set_ylabel("$\Delta h$ (mm)")
# ax1.set_title("Model Comparison")
plt.savefig("model_comparison_pres.png", dpi=fig.dpi)
plt.show()

