import cv2
import pickle, sys
import numpy as np
sys.path.append('../toolbox/')
from flir_toolbox import *
from angled_layers import *


# Load the IR recording data from the pickle file
with open('../../recorded_data/ER4043_bent_tube_2024_09_04_12_23_40/layer_39/ir_recording.pickle', 'rb') as file:
    ir_recording = pickle.load(file)

ir_ts=np.loadtxt('../../recorded_data/ER4043_bent_tube_2024_09_04_12_23_40/layer_39/ir_stamps.csv', delimiter=',')

print(len(ir_recording), len(ir_ts))

result = cv2.VideoWriter('output.mp4', 
                         cv2.VideoWriter_fourcc(*'H264'),
                         30, (240,320))

# Create a window to display the images
cv2.namedWindow("IR Recording", cv2.WINDOW_NORMAL)
# Set the colormap (inferno) and normalization range for the color bar
cmap = cv2.COLORMAP_INFERNO
colorbar_min = np.min(ir_recording)
colorbar_max = np.max(ir_recording)
colorbar_max_pos = np.max(ir_recording)
print(ir_recording.shape)

temp_max = [1300]
# counts2temp(np.array([colorbar_max]), 6.39661118e+03, 1.40469989e+03, 1.00000008e+00, 8.69393436e+00, 8.40029488e+03,Emiss=0.13)
print("Temp Range: ", temp_max)
x_min=50
x_max=-50
y_min=50
y_max=-50
for i in range(len(ir_recording)):
    # print(np.max(ir_recording[i]), np.min(ir_recording[i]))
    centroid, bbox=flame_detection_aluminum(ir_recording[i])

    temp=counts2temp(
        ir_recording[i].flatten(),
        6.39661118e+03,
        1.40469989e+03,
        1.00000008e+00,
        8.69393436e+00,
        8.40029488e+03,
        Emiss=0.13
    ).reshape((240,320))
    temp[temp > 1300] = 1300    ##thresholding
    # Normalize the data to [0, 255]
    ir_normalized = ((temp - 0) / (temp_max[0] - 0)) * 255

    # ir_normalized = ir_normalized[x_min:x_max, y_min:y_max]
    ir_normalized=np.clip(ir_normalized, 0, 255)
    # Convert the IR image to BGR format with the inferno colormap
    # print(ir_normalized.astype(np.uint8))
    ir_bgr = cv2.applyColorMap(ir_normalized.astype(np.uint8), cv2.COLORMAP_INFERNO)

    # add bounding box
    if centroid is not None:
        cv2.rectangle(ir_bgr, (bbox[0],bbox[1]), (bbox[0]+bbox[2],bbox[1]+bbox[3]), (0,255,0), thickness=1)   #flame bbox
        bbox_below_size=10
        centroid_below=(int(centroid[0]+bbox[2]/2+bbox_below_size/2),centroid[1])
        # cv2.rectangle(ir_bgr, (int(centroid_below[0]-bbox_below_size/2),int(centroid_below[1]-bbox_below_size/2)), (int(centroid_below[0]+bbox_below_size/2),int(centroid_below[1]+bbox_below_size/2)), (0,255,0), thickness=1)   #flame below centroid

    # cv2.rectangle(ir_bgr, (50,110,95,60), (255,0,0), thickness=1)   #flame below centroid

    # Write the IR image to the video file
    rotated=cv2.rotate(ir_bgr, cv2.ROTATE_90_CLOCKWISE)
    result.write(rotated)

    # Display the IR image
    cv2.imshow("IR Recording", rotated)

    # # Wait for a specific time (in milliseconds) before displaying the next frame
    cv2.waitKey(int(1000*(ir_ts[i+1]-ir_ts[i])))

result.release()
# Close the window after the loop is completed
cv2.destroyAllWindows()
