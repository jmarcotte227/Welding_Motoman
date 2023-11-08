import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
from scipy.spatial.transform import Rotation as R

sys.path.append('../../weld/')
import weld_dh2v

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy
########################################################################
# limits of welding bead height (based on min and Max estimate speed from Eric)
    # needs to be modified based on actual limits from Eric 
def main():
    min_speed = 5
    max_speed = 15
    feed_speed = 160
    material = 'ER_4043'

    max_dH = weld_dh2v.v2dh_loglog(min_speed,feed_speed,material)
    min_dH = weld_dh2v.v2dh_loglog(max_speed,feed_speed,material)
    mean_dH = (max_dH+min_dH)/2

    print('Max dH: ', max_dH)
    print('Min dH: ', min_dH)
    print('Mean dH: ', mean_dH)
    print('------------------------')

    #wall characteristics
    wall_length = 100
    points_distance=0.5
    num_layers = 30
    points_per_layer=int(wall_length/points_distance)


    #rotation criteria
    layer_angle = np.arcsin((max_dH-min_dH)/wall_length)
    rot_point = max_dH/np.tan(layer_angle)
    print('Layer Angle:', np.rad2deg(layer_angle))
    print('Final Angle:', np.rad2deg(layer_angle)*num_layers)
    print('Point of Rotation:', rot_point)
      
    


    #layer gen
    curve_curved=np.zeros((num_layers*points_per_layer,6))

    #first layer
    curve_curved[0:points_per_layer,0]=np.linspace(0,wall_length,points_per_layer)
    curve_curved[0:points_per_layer,2]=0
    curve_curved[0:points_per_layer,-1]=-np.ones(points_per_layer)


    for layer in range(num_layers-1):
        for point in range(points_per_layer):
              dx,dz = rotate([rot_point, 0], (curve_curved[layer*points_per_layer+point,0],curve_curved[layer*points_per_layer+point,2]),-layer_angle)
              curve_curved[(layer+1)*points_per_layer+point,0] = dx
              curve_curved[(layer+1)*points_per_layer+point,2] = dz

              grav_dx,grav_dz = rotate((0,0), (curve_curved[layer*points_per_layer+point,3],curve_curved[layer*points_per_layer+point,5]),-layer_angle)
              curve_curved[(layer+1)*points_per_layer+point,3] = grav_dx
              curve_curved[(layer+1)*points_per_layer+point,5] = grav_dz


    vis_step=20
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot3D(curve_curved[::vis_step,0],curve_curved[::vis_step,1],curve_curved[::vis_step,2],'r.-')
    ax.quiver(curve_curved[::vis_step,0],curve_curved[::vis_step,1],curve_curved[::vis_step,2],curve_curved[::vis_step,3],curve_curved[::vis_step,4],curve_curved[::vis_step,5],length=10, normalize=True)
    plt.show()

    for layer in range(num_layers):
	    np.savetxt('slice_ER_4043/curve_sliced/slice'+str(layer)+'_0.csv',curve_curved[layer*points_per_layer:(layer+1)*points_per_layer],delimiter=',')
        


if __name__ == '__main__':
      main()	
