import cv2
import pickle, os, inspect
import numpy as np
from flir_toolbox import *
from ultralytics import YOLO

# Load the IR recording data from the pickle file
# data_dir='../../../recorded_data/ER316L/wallbf_140ipm_v14_140ipm_v14/'
# data_dir='../../../recorded_data/ER316L/trianglebf_100ipm_v10_100ipm_v10/'
# data_dir='../../../recorded_data/ER316L/streaming/cylinderspiral_T25000/'
# data_dir='../../../recorded_data/ER316L/phi0.9_VPD20/cylinderspiral_180ipm_v9/'
# data_dir='../../../recorded_data/ER4043/wallbf_100ipm_v10_100ipm_v10/'
# data_dir='../../../recorded_data/wall_weld_test/4043_150ipm_2024_06_18_11_16_32/layer_8/'
# data_dir='../../../recorded_data/ER316L/VPD10/tubespiral_180ipm_v18/'
# data_dir='../../../recorded_data/ER316L/streaming/right_triangle/bf_ol_v10_f100/'
# data_dir='../../../recorded_data/ER4043_bent_tube/'
data_dir = f'../../../recorded_data/ER4043_bent_tube_2024_08_01_11_47_23/layer_30/'


torch_model = YOLO(os.path.dirname(inspect.getfile(flir_toolbox))+"/torch.pt")
tip_model = YOLO(os.path.dirname(inspect.getfile(flir_toolbox))+"/tip_wire.pt")

with open(data_dir+'/ir_recording.pickle', 'rb') as file:
    ir_recording = pickle.load(file)
ir_ts=np.loadtxt(data_dir+'/ir_stamps.csv', delimiter=',')

#load template
# template = cv2.imread('torch_template_ER316L.png',0)

# Create a window to display the images
cv2.namedWindow("IR Recording", cv2.WINDOW_NORMAL)

# Set the colormap (inferno) and normalization range for the color bar
cmap = cv2.COLORMAP_INFERNO

# Function to update the frame
def update_frame(val):
    i = cv2.getTrackbarPos('Frame', 'IR Recording')
    ir_image = np.rot90(ir_recording[i], k=-1)
    centroid, bbox, torch_centroid, torch_bbox=weld_detection_aluminum(ir_image,torch_model,percentage_threshold=0.8)
    #centroid, bbox, torch_centroid, torch_bbox=weld_detection_steel(ir_image,torch_model,tip_model)

    ir_normalized = ((ir_image - np.min(ir_image)) / (np.max(ir_image) - np.min(ir_image))) * 255
    ir_normalized=np.clip(ir_normalized, 0, 255)

    # Convert the IR image to BGR format with the inferno colormap
    ir_bgr = cv2.applyColorMap(ir_normalized.astype(np.uint8), cv2.COLORMAP_INFERNO)

    
    if centroid is not None:
        #bbox for 5x5 window below the centroid
        cv2.rectangle(ir_bgr, (int(centroid[0]-2), int(centroid[1])), (int(centroid[0]+2), int(centroid[1]+4)), (255,150,50), thickness=1)

        # cv2.rectangle(ir_bgr, (bbox[0],bbox[1]), (bbox[0]+bbox[2],bbox[1]+bbox[3]), (0,255,0), thickness=1)
        #draw a circle at the centroid
        cv2.circle(ir_bgr, (int(centroid[0]), int(centroid[1])), 1, (0,0,0), -1)
        
    
    # if torch_centroid is not None:
    #     cv2.rectangle(ir_bgr, (torch_bbox[0],torch_bbox[1]), (torch_bbox[0]+torch_bbox[2],torch_bbox[1]+torch_bbox[3]), (0,255,0), thickness=1)
        ###mark the bottom center of the torch
        # cv2.circle(ir_bgr, (int(torch_bbox[0]+torch_bbox[2]/2),torch_bbox[1]+torch_bbox[3]), 3, (0,255,0), -1)

    # Display the IR image
    cv2.imshow("IR Recording", ir_bgr)

# Create the trackbars
cv2.createTrackbar('Frame', 'IR Recording', 1, min(len(ir_recording),len(ir_ts)) - 1, update_frame)
cv2.createTrackbar('Play', 'IR Recording', 0, 1, lambda x: None)

# Initialize with the first frame
update_frame(0)

i = 1
while True:
    # Check if the 'Play' trackbar is set to 1 (play)
    if cv2.getTrackbarPos('Play', 'IR Recording') == 1:
        # Increment the frame index
        i = (i + 1) % len(ir_recording)

        # Update the 'Frame' trackbar position
        cv2.setTrackbarPos('Frame', 'IR Recording', i)
    
    else:
        # Update the frame index to the current trackbar position
        i = cv2.getTrackbarPos('Frame', 'IR Recording')

    ###Display the timestamp in Terminal
    print('\rTimeStamp: %.5f' %(ir_ts[i]-ir_ts[0]), end='', flush=True)

    timestep=max(20,int(1000*(ir_ts[i]-ir_ts[i-1])))
    # Wait for a specific time (in milliseconds) before displaying the next frame
    if cv2.waitKey(timestep) & 0xFF == ord('q'):
        break

# Close the window after the loop is completed
cv2.destroyAllWindows()