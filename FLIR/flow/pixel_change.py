import cv2
import pickle, sys
import numpy as np
sys.path.append('../../toolbox/')
from flir_toolbox import *

def normalize2cv(frame):
    ir_normalized = ((frame - np.min(frame)) / (np.max(frame) - np.min(frame))) * 255
    return ir_normalized.astype(np.uint8)


# Load the IR recording data from the pickle file
with open('../../../recorded_data/316L_model_120ipm_2023_09_25_19_56_43/layer_9/ir_recording.pickle', 'rb') as file:
    ir_recording = pickle.load(file)

ir_ts=np.loadtxt('../../../recorded_data/316L_model_120ipm_2023_09_25_19_56_43/layer_9/ir_stamps.csv', delimiter=',')

print(len(ir_recording), len(ir_ts))

result = cv2.VideoWriter('output.avi', 
                         cv2.VideoWriter_fourcc(*'XVID'),
                         30, (320,240))

# Create a window to display the images
cv2.namedWindow("IR Flow", cv2.WINDOW_NORMAL)
# Set the colormap (inferno) and normalization range for the color bar
cmap = cv2.COLORMAP_INFERNO
colorbar_min = np.min(ir_recording)
colorbar_max = np.max(ir_recording)

for i in range(1,len(ir_recording)):
    prev=ir_recording[i-1]
    cur=ir_recording[i]
    # Compute the frame difference
    # diff = cv2.absdiff(prev, cur)
    diff=cur.astype(np.double)-prev.astype(np.double)
    print(np.max(diff),np.min(diff))
    # Optionally apply a colormap to the difference
    diff_colored = cv2.applyColorMap(normalize2cv(diff), cv2.COLORMAP_INFERNO)
    
    result.write(diff_colored)
    # cv2.imshow("IR Flow",diff_colored)
    # # Wait for a specific time (in milliseconds) before displaying the next frame
    # cv2.waitKey(int(1000*(ir_ts[i+1]-ir_ts[i])))

result.release()
# Close the window after the loop is completed
cv2.destroyAllWindows()
