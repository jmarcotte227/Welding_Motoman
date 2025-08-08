import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../../../toolbox")
import angled_layers as al
from tqdm import tqdm

temps = []
# TEST_ID = 'ER4043_bent_tube_large_cold_2024_11_07_10_21_39'
# TEST_ID = 'ER4043_bent_tube_large_hot_2024_11_06_12_27_19'
# TEST_ID = 'ER4043_bent_tube_large_cold_OL_2024_11_14_11_56_43'
TEST_ID = 'ER4043_bent_tube_large_hot_OL_2024_11_14_13_05_38'
for layer in tqdm(range(1, 107)):
    recorded_dir = (
        f"../../../../recorded_data/{TEST_ID}/layer_{layer}/"
    )
    max_temp, avg_temp, min_temp, job_no = al.flame_temp(recorded_dir)
    job_no_offset = 3
    job_no = job_no - job_no_offset
    data = np.vstack((job_no, max_temp)).T
    data = data[(0 <= data[:, 0]) & (49 >= data[:, 0])]
    averages= al.avg_by_line(data[:,0], data[:,1], np.linspace(0,49,50))
    temps.append(averages)

temps=np.squeeze(np.array(temps))
print(temps.shape)
# np.savetxt(f"{TEST_ID}_temps.csv", temps,delimiter=',')
np.savetxt(f"{TEST_ID}_max_temps.csv", temps,delimiter=',')

# with open(f"ER4043_bent_tube_large_hot_2024_11_06_12_27_19_temps.pkl", 'wb') as file:
#     pickle.dump(temps, file)
