import cv2
import pickle, sys
import numpy as np
sys.path.append('../toolbox/')
from flir_toolbox import *

# Load the IR recording data from the pickle file
# data_dir='../../recorded_data/ER316L/wallbf_70ipm_v7_70ipm_v7/'
# data_dir='../../recorded_data/ER316L/trianglebf_100ipm_v10_100ipm_v10/'
# data_dir='../../recorded_data/trianglebf_100ipm_v10_100ipm_v10/'
# data_dir='../../recorded_data/wall_weld_test/5356_150ipm_2024_06_17_14_36_16/layer_2'
data_dir='../../recorded_data/ER4043_bent_tube_2024_08_22_11_12_27/layer_50'
with open(data_dir+'/ir_recording.pickle', 'rb') as file:
    ir_recording = pickle.load(file)
ir_ts=np.loadtxt(data_dir+'/ir_stamps.csv', delimiter=',')


# Create a window to display the images
cv2.namedWindow("IR Recording", cv2.WINDOW_NORMAL)

# Set the colormap (inferno) and normalization range for the color bar
cmap = cv2.COLORMAP_INFERNO

# Function to update the frame
i = 100
ir_image = np.rot90(ir_recording[i], k=-1)

ir_normalized = ((ir_image - np.min(ir_image)) / (np.max(ir_image) - np.min(ir_image))) * 255
ir_normalized=np.clip(ir_normalized, 0, 255)

# Convert the IR image to BGR format with the inferno colormap
ir_bgr = cv2.applyColorMap(ir_normalized.astype(np.uint8), cv2.COLORMAP_INFERNO)

# Display the IR image
cv2.imshow("IR Recording", ir_bgr)
