import numpy as np

for layer in range(40,60):
    curr_layer = np.loadtxt(f"slice/curve_sliced/slice{layer+1}_0.csv", delimiter = ',')
    next_layer = np.loadtxt(f"slice/curve_sliced/slice{layer+2}_0.csv", delimiter = ',')

    min = np.linalg.norm(curr_layer[0,:3]-next_layer[0, :3])
    max = np.linalg.norm(curr_layer[-1,:3]-next_layer[-1, :3])
    print(f"Layer {layer+1}, max : {max:.3f}, min: {min:.3f}")

