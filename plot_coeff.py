import numpy as np
import matplotlib.pyplot as plt
import glob
data_dir = f"../recorded_data/test_dir/"
folders = glob.glob(data_dir+'layer_*/coeff_mat.csv')

coeff_mats = []

for folder in folders:
    mat = np.loadtxt(folder, delimiter = ',')
    coeff_mats.append(mat)

coeff_mats = np.array(coeff_mats)
print(coeff_mats.shape)

plt.plot(coeff_mats[:,0])
plt.plot(coeff_mats[:,1])
plt.show()
