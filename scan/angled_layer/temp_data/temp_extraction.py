import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../../../toolbox")
import angled_layers as al

temps = []
for layer in range(1, 107):
    recorded_dir = (
        f"../../../../recorded_data/ER4043_bent_tube_large_hot_2024_11_06_12_27_19/layer_{layer}/"
    )
    max_temp, avg_temp, min_temp, job_no = al.flame_temp(recorded_dir)
    job_no_offset = 3
    job_no = job_no - job_no_offset
    data = np.vstack((job_no, avg_temp)).T
    data = data[(0 <= data[:, 0]) & (49 >= data[:, 0])]

    temps.append(data)

with open(f"ER4043_bent_tube_large_hot_2024_11_06_12_27_19_temps.pkl", 'wb') as file:
    pickle.dump(temps, file)
