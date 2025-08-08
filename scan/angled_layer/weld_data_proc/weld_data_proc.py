import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../../../toolbox")
import angled_layers as al
from tqdm import tqdm

def weld_param_jobno(save_path):
    '''
    Takes the welding parameters, correlates them to job line numbers.
    '''
    weld_data = np.loadtxt(save_path + "welding.csv", delimiter=",", skiprows=1)
    joint_angle = np.loadtxt(save_path + "weld_js_exe.csv", delimiter=",")

    job_no = []

    # find all pixel regions to record from flame detection
    for i in range(weld_data.shape[0]):
        joint_idx = np.argmin(np.abs(weld_data[i,0] - joint_angle[:, 0]))
        job_no.append(int(joint_angle[joint_idx][1]))

    job_no = np.array(job_no)
    return weld_data, job_no

# TEST_ID = 'ER4043_bent_tube_large_cold_2024_11_07_10_21_39'
# TEST_ID = 'ER4043_bent_tube_large_hot_2024_11_06_12_27_19'
TEST_ID = 'ER4043_bent_tube_large_cold_OL_2024_11_14_11_56_43'
# TEST_ID = 'ER4043_bent_tube_large_hot_OL_2024_11_14_13_05_38'

voltages = []
currents = []
feedrates = []
energy = []

for layer in tqdm(range(1, 107)):
    recorded_dir = (
        f"../../../../recorded_data/{TEST_ID}/layer_{layer}/"
    )
    weld_data, job_no = weld_param_jobno(recorded_dir)
    job_no_offset = 3
    job_no = job_no - job_no_offset
    job_no = np.expand_dims(job_no,axis=1)
    data = np.hstack((job_no, weld_data))
    data = np.array(data[(0 <= data[:, 0]) & (49 >= data[:, 0])])
    voltages.append(al.avg_by_line(data[:,0], data[:,2], np.linspace(0,49,50)))
    currents.append(al.avg_by_line(data[:,0], data[:,3], np.linspace(0,49,50)))
    feedrates.append(al.avg_by_line(data[:,0], data[:,4], np.linspace(0,49,50)))
    energy.append(al.avg_by_line(data[:,0], data[:,5], np.linspace(0,49,50)))

voltages = np.squeeze(np.array(voltages))
currents = np.squeeze(np.array(currents))
feedrates = np.squeeze(np.array(feedrates))
energy = np.squeeze(np.array(energy))

np.savetxt(f"{TEST_ID}_voltages.csv", voltages, delimiter=',')
np.savetxt(f"{TEST_ID}_currents.csv", currents, delimiter=',')
np.savetxt(f"{TEST_ID}_feedrates.csv", feedrates, delimiter=',')
np.savetxt(f"{TEST_ID}_energy.csv", energy, delimiter=',')

# with open(f"ER4043_bent_tube_large_hot_2024_11_06_12_27_19_temps.pkl", 'wb') as file:
#     pickle.dump(temps, file)
