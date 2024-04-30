import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
from scipy.spatial.transform import Rotation as R
import pickle

sys.path.append('../../weld/')
import weld_dh2v

########################################################################
# limits of welding bead height (based on min and Max estimate speed from Eric)
    # needs to be modified based on actual limits from Eric 
def main():
    feed_speeds = [160,200]
    torch_speeds = [13, 13.75]
    material = 'ER_4043'

    dH = weld_dh2v.v2dh_loglog(torch_speeds[0],feed_speeds[0],material)
    # for i in range(1,len(feed_speeds)): 
    #       print(i)
    #       torch_speeds.append(weld_dh2v.dh2v_loglog(dH,feed_speeds[i],material))


    print('Torch Speeds: ', torch_speeds)
    print('Feed Speed: ', feed_speeds)
    print('dH: ', dH)
    print('------------------------')

    #wall characteristics
    wall_length = 100
    points_distance=0.5
    num_layers = 31
    points_per_layer=int(wall_length/points_distance)
    vertical_shift = 6 #mm

    #layer gen
    curve_curved=np.zeros((num_layers*points_per_layer,6))
    base_layer = np.zeros((points_per_layer,6))

    #base layer
    base_layer[0:points_per_layer,0]=np.linspace(0,wall_length,points_per_layer)
    base_layer[0:points_per_layer,2]=0
    base_layer[0:points_per_layer,-1]=-np.ones(points_per_layer)

    np.savetxt('slice_ER4043_160_200/curve_sliced/slice0_0.csv',base_layer,delimiter=',')

    base_layer[0:points_per_layer,0]=np.linspace(0,wall_length,points_per_layer)
    base_layer[0:points_per_layer,2]=3
    base_layer[0:points_per_layer,-1]=-np.ones(points_per_layer)

    np.savetxt('slice_ER4043_160_200/curve_sliced/slice0_1.csv',base_layer,delimiter=',')
    #first layer
    curve_curved[0:points_per_layer,0]=np.linspace(0,wall_length,points_per_layer)
    curve_curved[0:points_per_layer,2]=vertical_shift
    curve_curved[0:points_per_layer,-1]=-np.ones(points_per_layer)


    for layer in range(num_layers-1):
        for point in range(points_per_layer):
              #print(dx,dz)
              curve_curved[(layer+1)*points_per_layer+point,0] = curve_curved[(layer)*points_per_layer+point,0]
              curve_curved[(layer+1)*points_per_layer+point,2] = curve_curved[(layer)*points_per_layer+point,2]+dH

              curve_curved[(layer+1)*points_per_layer+point,3] = curve_curved[(layer)*points_per_layer+point,3]
              curve_curved[(layer+1)*points_per_layer+point,5] = curve_curved[(layer)*points_per_layer+point,5]



    vis_step=1
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot3D(curve_curved[::vis_step,0],curve_curved[::vis_step,1],curve_curved[::vis_step,2],'r.-')
    ax.quiver(curve_curved[::vis_step,0],curve_curved[::vis_step,1],curve_curved[::vis_step,2],curve_curved[::vis_step,3],curve_curved[::vis_step,4],curve_curved[::vis_step,5],length=10, normalize=True)
    plt.show()

    for layer in range(num_layers):
        np.savetxt('slice_ER4043_160_200_comp/curve_sliced/slice'+str(layer+1)+'_0.csv',curve_curved[layer*points_per_layer:(layer+1)*points_per_layer],delimiter=',')
    
    vel_profile = np.zeros(points_per_layer)  
    feed_profile = np.zeros(points_per_layer)
    zones = len(feed_speeds)
    points_per_zone = int(points_per_layer/zones)
    print(points_per_zone)
    for i in range(zones):
     for j in range(points_per_zone):
         vel_profile[int(i*points_per_zone+j)] = torch_speeds[i]
         feed_profile[int(i*points_per_zone+j)] = feed_speeds[i]
    print(vel_profile)
    print(feed_profile)
    
    with open('slice_ER4043_160_200_comp/vel_profile.pkl', 'wb') as file:
         pickle.dump(vel_profile, file)
    with open('slice_ER4043_160_200_comp/feed_profile.pkl', 'wb') as file:
         pickle.dump(feed_profile, file)

if __name__ == '__main__':
      main()	
