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
         "seg1a": [200, 4.48],
         "seg1b": [200, 3.81],
         "seg1f": [240, 6.08],
         "seg2a": [200, 4.48],
         "seg2b": [240, 7.16],
         "seg2f": [240, 6.08],
         "seg3a": [200, 4.48],
         "seg3b": [200, 4.48],
         "seg3f": [240, 6.09],
         "seg4a": [200, 3.81],
         "seg4b": [200, 3.81],
         "seg4f": [240, 6.08],
    }

    

    #segment characteristics
    num_seg = 12 #should match bead_params
    seg_len = 40 #mm
    points_distance=0.5
    num_layers = 2
    gap = 40
    groups = 2

    #base_seg params
    num_pre_seg = 2
    base_seg_len = 100
    base_seg_min = 0
    base_seg_max = 80
    work_offset = 20

    ####CHANGE AFTER INITIAL TEST TOMORROW
    x1 = 20
    d1 = 8.26

    x2 = 40
    d2 = 9.40
    
    x3 = 60
    d3 = 8.39

    x4 = 80
    d4 = 8.15


    points_per_segment=int(seg_len/points_distance)
    points_per_base_seg = int(base_seg_len/points_distance)
    points_per_layer = points_per_segment*num_seg+points_per_base_seg*num_pre_seg
    vertical_shift = 0 #mm add 3 for base layer
    
    #layer gen
    curve_curved=np.zeros((num_layers*points_per_layer,6))
 

    #first layer


    for layer in range(1):
        #first baseseg
        curve_curved[0:points_per_base_seg, 0] = np.linspace(0, base_seg_len, points_per_base_seg)
        curve_curved[0:points_per_base_seg, 1] = base_seg_min
        curve_curved[0:points_per_base_seg, -1] = -1

        #second baseseg
        curve_curved[points_per_base_seg:points_per_base_seg*2, 0] = np.linspace(0, base_seg_len, points_per_base_seg)
        curve_curved[points_per_base_seg:points_per_base_seg*2, 1] = base_seg_max
        curve_curved[points_per_base_seg:points_per_base_seg*2, -1] = -1
    
        base_offset = points_per_base_seg*2
             
        #first group
            #seg1a
        seg_idx = 0
        curve_curved[base_offset+seg_idx*points_per_segment:base_offset+(seg_idx+1)*points_per_segment, 0] = x1
        curve_curved[base_offset+seg_idx*points_per_segment:base_offset+(seg_idx+1)*points_per_segment, 1] = np.linspace(work_offset, work_offset+seg_len, points_per_segment)
        curve_curved[base_offset+seg_idx*points_per_segment:base_offset+(seg_idx+1)*points_per_segment, -1] = -1
            #seg1b
        seg_idx = 1
        curve_curved[base_offset+seg_idx*points_per_segment:base_offset+(seg_idx+1)*points_per_segment, 0] = d1+x1
        curve_curved[base_offset+seg_idx*points_per_segment:base_offset+(seg_idx+1)*points_per_segment, 1] = np.linspace(work_offset, work_offset+seg_len, points_per_segment)
        curve_curved[base_offset+seg_idx*points_per_segment:base_offset+(seg_idx+1)*points_per_segment, -1] = -1  
            #seg1f   
        seg_idx = 2
        curve_curved[base_offset+seg_idx*points_per_segment:base_offset+(seg_idx+1)*points_per_segment, 0] = d1/2+x1
        curve_curved[base_offset+seg_idx*points_per_segment:base_offset+(seg_idx+1)*points_per_segment, 1] = np.linspace(work_offset, work_offset+seg_len, points_per_segment)
        curve_curved[base_offset+seg_idx*points_per_segment:base_offset+(seg_idx+1)*points_per_segment, -1] = -1

        #2nd group
            #seg2a
        seg_idx = 3
        curve_curved[base_offset+seg_idx*points_per_segment:base_offset+(seg_idx+1)*points_per_segment, 0] = x2
        curve_curved[base_offset+seg_idx*points_per_segment:base_offset+(seg_idx+1)*points_per_segment, 1] = np.linspace(work_offset, work_offset+seg_len, points_per_segment)
        curve_curved[base_offset+seg_idx*points_per_segment:base_offset+(seg_idx+1)*points_per_segment, -1] = -1
            #seg2b
        seg_idx = 4
        curve_curved[base_offset+seg_idx*points_per_segment:base_offset+(seg_idx+1)*points_per_segment, 0] = d2+x2
        curve_curved[base_offset+seg_idx*points_per_segment:base_offset+(seg_idx+1)*points_per_segment, 1] = np.linspace(work_offset, work_offset+seg_len, points_per_segment)
        curve_curved[base_offset+seg_idx*points_per_segment:base_offset+(seg_idx+1)*points_per_segment, -1] = -1  
            #seg2f   
        seg_idx = 5
        curve_curved[base_offset+seg_idx*points_per_segment:base_offset+(seg_idx+1)*points_per_segment, 0] = d2/2+x2
        curve_curved[base_offset+seg_idx*points_per_segment:base_offset+(seg_idx+1)*points_per_segment, 1] = np.linspace(work_offset, work_offset+seg_len, points_per_segment)
        curve_curved[base_offset+seg_idx*points_per_segment:base_offset+(seg_idx+1)*points_per_segment, -1] = -1

        #3nd group
            #seg3a
        seg_idx = 6
        curve_curved[base_offset+seg_idx*points_per_segment:base_offset+(seg_idx+1)*points_per_segment, 0] = x3
        curve_curved[base_offset+seg_idx*points_per_segment:base_offset+(seg_idx+1)*points_per_segment, 1] = np.linspace(work_offset, work_offset+seg_len, points_per_segment)
        curve_curved[base_offset+seg_idx*points_per_segment:base_offset+(seg_idx+1)*points_per_segment, -1] = -1
            #seg3b
        seg_idx = 7
        curve_curved[base_offset+seg_idx*points_per_segment:base_offset+(seg_idx+1)*points_per_segment, 0] = d3+x3
        curve_curved[base_offset+seg_idx*points_per_segment:base_offset+(seg_idx+1)*points_per_segment, 1] = np.linspace(work_offset, work_offset+seg_len, points_per_segment)
        curve_curved[base_offset+seg_idx*points_per_segment:base_offset+(seg_idx+1)*points_per_segment, -1] = -1  
            #seg3f   
        seg_idx = 8
        curve_curved[base_offset+seg_idx*points_per_segment:base_offset+(seg_idx+1)*points_per_segment, 0] = d3/2+x3
        curve_curved[base_offset+seg_idx*points_per_segment:base_offset+(seg_idx+1)*points_per_segment, 1] = np.linspace(work_offset, work_offset+seg_len, points_per_segment)
        curve_curved[base_offset+seg_idx*points_per_segment:base_offset+(seg_idx+1)*points_per_segment, -1] = -1

        #4nd group
            #seg4a
        seg_idx = 9
        curve_curved[base_offset+seg_idx*points_per_segment:base_offset+(seg_idx+1)*points_per_segment, 0] = x4
        curve_curved[base_offset+seg_idx*points_per_segment:base_offset+(seg_idx+1)*points_per_segment, 1] = np.linspace(work_offset, work_offset+seg_len, points_per_segment)
        curve_curved[base_offset+seg_idx*points_per_segment:base_offset+(seg_idx+1)*points_per_segment, -1] = -1
            #seg4b
        seg_idx = 10
        curve_curved[base_offset+seg_idx*points_per_segment:base_offset+(seg_idx+1)*points_per_segment, 0] = d4+x4
        curve_curved[base_offset+seg_idx*points_per_segment:base_offset+(seg_idx+1)*points_per_segment, 1] = np.linspace(work_offset, work_offset+seg_len, points_per_segment)
        curve_curved[base_offset+seg_idx*points_per_segment:base_offset+(seg_idx+1)*points_per_segment, -1] = -1  
            #seg4f   
        seg_idx = 11
        curve_curved[base_offset+seg_idx*points_per_segment:base_offset+(seg_idx+1)*points_per_segment, 0] = d4/2+x4
        curve_curved[base_offset+seg_idx*points_per_segment:base_offset+(seg_idx+1)*points_per_segment, 1] = np.linspace(work_offset, work_offset+seg_len, points_per_segment)
        curve_curved[base_offset+seg_idx*points_per_segment:base_offset+(seg_idx+1)*points_per_segment, -1] = -1
    #duplicate with offset
    curve_curved[points_per_layer: points_per_layer*2, :] = curve_curved[0: +points_per_layer, :]
    vis_step=1
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.set_aspect("equal")
    ax.plot3D(curve_curved[::vis_step,0],curve_curved[::vis_step,1],curve_curved[::vis_step,2],'r.-')
    #ax.quiver(curve_curved[::vis_step,0],curve_curved[::vis_step,1],curve_curved[::vis_step,2],curve_curved[::vis_step,3],curve_curved[::vis_step,4],curve_curved[::vis_step,5],length=10, normalize=True)
    ax.set_aspect("equal")
    plt.show()

    for layer in range(num_layers):
         for seg in range(num_pre_seg):
              np.savetxt('curve_sliced/slice'+str(layer)+'_'+str(seg)+'.csv',
                     curve_curved[layer*points_per_layer+seg*points_per_base_seg:layer*points_per_layer+(1+seg)*points_per_base_seg],delimiter=',')
         for seg in range(num_seg):
             np.savetxt('curve_sliced/slice'+str(layer)+'_'+str(seg+num_pre_seg)+'.csv',
                     curve_curved[base_offset+layer*points_per_layer+seg*points_per_segment:base_offset+layer*points_per_layer+(1+seg)*points_per_segment],delimiter=',')
    with open('bead_params.pkl', 'wb') as file:
        pickle.dump(bead_params, file)    


if __name__ == '__main__':
      main()	
