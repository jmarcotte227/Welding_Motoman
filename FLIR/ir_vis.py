import numpy as np
import matplotlib.pyplot as plt
import pickle, sys
sys.path.append('../toolbox/')
from flir_toolbox import *

with open('../../recorded_data/ER4043_bent_tube_2024_09_04_12_23_40/layer_2/ir_recording.pickle', 'rb') as file:
    ir_recording = pickle.load(file)
freq=30
print(type(ir_recording))
fig = plt.figure(1)
for i in range(len(ir_recording)):
    temp=counts2temp(ir_recording[i].flatten(),6.39661118e+03, 1.40469989e+03, 1.00000008e+00, 8.69393436e+00, 8.40029488e+03,Emiss=0.13).reshape((240,320))
    print(np.max(temp),np.min(temp))
    # temp_hdr=np.log(1 + temp) / np.log(1000)
    temp[temp > 1300] = 1300
    plt.imshow(temp, cmap='inferno', aspect='auto')
    plt.colorbar(format='%.2f')
    plt.pause(1/freq)
    plt.clf()
