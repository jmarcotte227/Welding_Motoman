import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../../toolbox")
import angled_layers as al

# moving average parameters
n = 5
n_offset = int((5 - 1) / 2)
# recorded_dirs = [
#         '../../../recorded_data/ER4043_bent_tube_2024_09_04_12_23_40/layer_4/',
#         '../../../recorded_data/ER4043_bent_tube_2024_09_04_12_23_40/layer_45/',
#         '../../../recorded_data/ER4043_bent_tube_2024_09_04_12_23_40/layer_70/',
# ]
# colors = ['r','g','b']
# fig,ax = plt.subplots(1,1)
# for idx, recorded_dir in enumerate(recorded_dirs):
#     max_temp, avg_temp, min_temp, job_no = al.flame_temp(recorded_dir)
#     job_no_offset = 3
#     job_no = job_no-job_no_offset
#     avg_max = al.avg_by_line(job_no, max_temp, np.linspace(0,49,50))
#     avg_avg = al.avg_by_line(job_no, avg_temp, np.linspace(0,49,50))
#     avg_min = al.avg_by_line(job_no, min_temp, np.linspace(0,49,50))
#     ax.scatter(np.linspace(1,50,50), avg_max, color=colors[idx])
#     ax.scatter(np.linspace(1,50,50), avg_avg, color=colors[idx], marker='s')
#     if idx==2: ax.scatter(np.linspace(1,50,50), avg_min, color = 'orange')

# ax.set_xlabel('Segment Index')
# ax.set_ylabel('Flame Pixel Value')
# ax.set_title('Pixel Value Across Chosen Layers')
# ax.legend([
#     'Layer 4 Max',
#     'Layer 4 Avg',
#     'Layer 45 Max',
#     'Layer 45 Avg',
#     'Layer 70 Max',
#     'Layer 70 Avg',
#     'Minimum Pixel Value'
# ])
# plt.show()

avg_temp_layer = []
med_temp_layer = []
for layer in range(1, 81):
    recorded_dir = (
        f"../../../recorded_data/ER4043_bent_tube_2024_09_04_12_23_40/layer_{layer}/"
    )
    max_temp, avg_temp, min_temp, job_no = al.flame_temp(recorded_dir)
    job_no_offset = 3
    job_no = job_no - job_no_offset
    data = np.vstack((job_no, avg_temp)).T
    data = data[(0 <= data[:, 0]) & (49 >= data[:, 0])]
    # avg_max = al.avg_by_line(job_no, max_temp, np.linspace(0,49,50))
    # avg_avg = al.avg_by_line(job_no, avg_temp, np.linspace(0,49,50))
    # avg_min = al.avg_by_line(job_no, min_temp, np.linspace(0,49,50))
    med_temp_layer.append(np.nanmedian(data[:, 1]))
    avg_temp_layer.append(np.nanmean(data[:, 1]))
    print(f"Layer {layer} Average: {avg_temp_layer[-1]}")
    print(f"Layer {layer} Median: {med_temp_layer[-1]}")
x = np.linspace(1, 80, 80)
x2 = x[2:-2]
wind_avg = np.convolve(med_temp_layer, np.ones(n) / n, mode="valid")
fig, ax = plt.subplots(1, 1)

ax.plot(x, med_temp_layer)
ax.plot(x, avg_temp_layer)
ax.plot(x2, wind_avg, "-r")
ax.legend(["Median Layer Temp", "Average Layer Temp"])
ax.set_title("Temp Across All Layers: Flame Max")
ax.set_xlabel("Layer Number")
ax.set_ylabel("Pixel Value")
plt.show()
