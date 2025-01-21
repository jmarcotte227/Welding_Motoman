import numpy as np

SQUARE_LEN = 100 #mm
SQUARE_WIDTH = 50 #mm
BASE_OFFSET = 200 #mm

verticies = np.array([
        [0, 0, 0],
        [SQUARE_LEN, 0, 0],
        [SQUARE_LEN, SQUARE_WIDTH, 0],
        [0, SQUARE_WIDTH, 0],
        [0, 0, 0]
    ])

base_layer = np.zeros((verticies.shape[0],6))
first_layer = np.zeros((verticies.shape[0],6))
for i in range(verticies.shape[0]):
    base_layer[i,0] = verticies[i,0]
    base_layer[i,1] = verticies[i,1]
    base_layer[i,2] = verticies[i,2]
    base_layer[i,-1] = -1


for i in range(verticies.shape[0]):
    first_layer[i,0] = verticies[i,0]
    first_layer[i,1] = verticies[i,1]
    first_layer[i,2] = verticies[i,2]+BASE_OFFSET
    first_layer[i,-1] = -1 

# save data 
np.savetxt('slice/curve_sliced/slice0_0.csv', base_layer, delimiter=',')
np.savetxt('slice/curve_sliced/slice1_0.csv', first_layer, delimiter=',')
