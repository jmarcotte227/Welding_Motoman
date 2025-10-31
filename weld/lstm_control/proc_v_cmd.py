import numpy as np
import matplotlib.pyplot as plt
max = 0

fig, ax = plt.subplots(1,1)
for layer in range(13):
    REC_DATA = '../../../recorded_data/wall_lstm_control_2025_10_28_16_02_17/'

    v_cmd = np.loadtxt(f"{REC_DATA}layer_{layer}/v_cmd.csv", delimiter=",")

    js_exe = np.loadtxt(f"{REC_DATA}layer_{layer}/weld_js_cmd.csv", delimiter=",")

    job_no = np.linspace(0,48, 49)

    v_cmds = []
    for num in job_no:
        idx = np.where(js_exe[:,1]==num)[0][0]
        v_cmds.append(v_cmd[idx+1])

    ax.plot(v_cmds)
ax.set_xlabel("Segment Index")
ax.set_ylabel("V_cmd (mm/s)")
plt.show()

