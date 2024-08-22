import numpy as np
import matplotlib.pyplot as plt
import glob
data_dir = f"../recorded_data/ER4043_bent_tube_2024_08_22_11_12_27/"
folders = glob.glob(data_dir+'layer_*/coeff_mat.csv')

coeff_mats = []

for folder in folders:
    print(folder)
    mat = np.loadtxt(folder, delimiter = ',')
    coeff_mats.append(mat)

coeff_mats = np.array(coeff_mats)
print(coeff_mats.shape)
x = np.linspace(0, coeff_mats.shape[0], coeff_mats.shape[0])    
plt.scatter(x, coeff_mats[:,0])
plt.scatter(x, coeff_mats[:,1])
plt.gca().grid()

plt.show()
