import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('../../toolbox')
import angled_layers as al

recorded_dirs = [
        '../../../recorded_data/ER4043_bent_tube_2024_09_04_12_23_40/layer_4/',
        '../../../recorded_data/ER4043_bent_tube_2024_09_04_12_23_40/layer_45/',
        '../../../recorded_data/ER4043_bent_tube_2024_09_04_12_23_40/layer_70/',
]
colors = ['r','g','b']
for idx, recorded_dir in enumerate(recorded_dirs):
    max_temp, avg_temp, min_temp, job_no = al.flame_temp(recorded_dir)
    job_no_offset = 3
    job_no = job_no-job_no_offset
    avg_max = al.avg_by_line(job_no, max_temp, np.linspace(0,49,50))
    avg_avg = al.avg_by_line(job_no, avg_temp, np.linspace(0,49,50))
    avg_min = al.avg_by_line(job_no, min_temp, np.linspace(0,49,50))
    plt.scatter(np.linspace(1,50,50), avg_max, color=colors[idx])
    # plt.scatter(np.linspace(1,50,50), avg_avg)
    plt.scatter(np.linspace(1,50,50), avg_min, color = 'orange')
plt.show()
