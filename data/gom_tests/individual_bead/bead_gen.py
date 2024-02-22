import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
from scipy.spatial.transform import Rotation as R
import pickle

sys.path.append('../../../weld/')
import weld_dh2v
import gom_funcs


def main():
    bead_params = {
         #segnum: [Vw, Vt]
         "base_seg1": [300, 5],
         "base_seg2": [300, 5],      
         "seg1": [200, 7.0],
         "seg2": [200, 6.5],
         "seg3": [200, 6.0],
         "seg4": [200, 5.5],
         "seg5": [200, 5.0],
         "seg6": [200, 4.5],
         "seg7": [200, 4.0],
         "seg8": [200, 3.5],
         "seg9": [200, 3.0],
    }

    

    #segment characteristics
    num_seg = 9 #should match bead_params
    seg_len = 30 #mm
    points_distance=0.5
    num_layers = 2
    row_offset = 40
    col_offset = 15
    num_rows = 3
    num_columns = 3

    #base_seg params
    num_pre_seg = 2
    base_seg_len = row_offset*2+seg_len
    base_seg_min = -col_offset
    base_seg_max = col_offset*3

    
    


    points_per_segment=int(seg_len/points_distance)
    points_per_base_seg = int(base_seg_len/points_distance)
    points_per_layer = points_per_segment*num_seg+points_per_base_seg*num_pre_seg
    vertical_shift = 0 #mm add 3 for base layer
    
    #layer gen
    curve_curved=np.zeros((num_layers*points_per_layer,6))
 

    #first layer


    for layer in range(num_layers):
        #first baseseg
        curve_curved[layer*points_per_layer:points_per_base_seg+layer*points_per_layer, 0] = np.linspace(0, base_seg_len, points_per_base_seg)
        curve_curved[layer*points_per_layer:points_per_base_seg+layer*points_per_layer, 1] = base_seg_min
        curve_curved[layer*points_per_layer:points_per_base_seg+layer*points_per_layer, 2] = layer*30
        curve_curved[layer*points_per_layer:points_per_base_seg+layer*points_per_layer, -1] = -1

        #second baseseg
        curve_curved[layer*points_per_layer+points_per_base_seg:layer*points_per_layer+points_per_base_seg*2, 0] = np.linspace(0, base_seg_len, points_per_base_seg)
        curve_curved[layer*points_per_layer+points_per_base_seg:layer*points_per_layer+points_per_base_seg*2, 1] = base_seg_max
        curve_curved[layer*points_per_layer+points_per_base_seg:layer*points_per_layer+points_per_base_seg*2, 2] = layer*30
        curve_curved[layer*points_per_layer+points_per_base_seg:layer*points_per_layer+points_per_base_seg*2, -1] = -1    
        
        base_offset = points_per_base_seg*2
             
        row_idx = 0
        col_idx = 0
        for segment in range(num_seg):
            curve_curved[layer*points_per_layer+base_offset+segment*points_per_segment:layer*points_per_layer+base_offset+(segment+1)*points_per_segment, 0] = np.linspace(row_idx*row_offset, row_idx*row_offset+seg_len, points_per_segment)
            curve_curved[layer*points_per_layer+base_offset+segment*points_per_segment:layer*points_per_layer+base_offset+(segment+1)*points_per_segment, 1] = col_offset*col_idx
            curve_curved[layer*points_per_layer+base_offset+segment*points_per_segment:layer*points_per_layer+base_offset+(segment+1)*points_per_segment, 2] = layer*10
            curve_curved[layer*points_per_layer+base_offset+segment*points_per_segment:layer*points_per_layer+base_offset+(segment+1)*points_per_segment, -1] = -1
            row_idx +=1
            if row_idx>=num_rows:
                 row_idx=0
                 col_idx+=1
             


    vis_step=1
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.set_aspect("equal")
    ax.plot3D(curve_curved[::vis_step,0],curve_curved[::vis_step,1],curve_curved[::vis_step,2],'r.-')
    #ax.quiver(curve_curved[::vis_step,0],curve_curved[::vis_step,1],curve_curved[::vis_step,2],curve_curved[::vis_step,3],curve_curved[::vis_step,4],curve_curved[::vis_step,5],length=10, normalize=True)
    ax.set_aspect("equal")
    plt.show()

    # for layer in range(num_layers):
    #      for seg in range(num_pre_seg):
    #           np.savetxt('curve_sliced/slice'+str(layer)+'_'+str(seg)+'.csv',
    #                  curve_curved[layer*points_per_layer+seg*points_per_base_seg:layer*points_per_layer+(1+seg)*points_per_base_seg],delimiter=',')
    #      for seg in range(num_seg):
    #          np.savetxt('curve_sliced/slice'+str(layer)+'_'+str(seg+num_pre_seg)+'.csv',
    #                  curve_curved[base_offset+layer*points_per_layer+seg*points_per_segment:base_offset+layer*points_per_layer+(1+seg)*points_per_segment],delimiter=',')
    with open('bead_params.pkl', 'wb') as file:
        pickle.dump(bead_params, file)    


if __name__ == '__main__':
      main()	
