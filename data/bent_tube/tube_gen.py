import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
from scipy.spatial.transform import Rotation as R

sys.path.append('../../weld/')
sys.path.append('../../toolbox/')
import angled_layers as al


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

def PointsInCircum(r,n):
    return [(np.cos(2*np.pi/n*x)*r,np.sin(2*np.pi/n*x)*r) for x in range(0,n+1)]
########################################################################
# limits of welding bead height (based on min and Max estimate speed from Eric)
    # needs to be modified based on actual limits from Eric 
def main():
    min_speed = 5
    max_speed = 15
    min_dH = 1.486
    max_dH = 1.859
    feed_speed = 160
    material = 'ER_4043'
    div_factor = 1 # change the angle between the bounds without changing the por

    # Initialize height-speed model
    model = al.SpeedHeightModel(a=-0.36997977, b=1.21532975)
    modelc = al.SpeedHeightModel()


    max_dH = model.v2dh(min_speed)/div_factor
    min_dH = modelc.v2dh(max_speed)/div_factor
    mean_dH = (max_dH+min_dH)/2

    print('Max dH: ', max_dH)
    print('Min dH: ', min_dH)
    print('Mean dH: ', mean_dH)
    print('Min vel: ', model.dh2v(max_dH))
    print('Max vel: ', model.dh2v(min_dH))
    print('------------------------')

    #tube characteristics
    tube_diameter = 50
    num_layers = 106
    points_per_layer=50
    point_distance = np.pi*tube_diameter/points_per_layer
    vertical_shift = 4 #mm  

    slices_per_layer = 1

    print('Point distance: ', point_distance)
    print('------------------------')


    #rotation criteria
    layer_angle = np.arcsin((max_dH-min_dH)/tube_diameter)
    rot_point = max_dH/np.tan(layer_angle)-tube_diameter/2
    layer_angle = layer_angle/slices_per_layer
    print('Layer Angle:', np.rad2deg(layer_angle))
    print('Final Angle:', np.rad2deg(layer_angle)*num_layers*slices_per_layer)
    print('Point of Rotation:', rot_point)
      
    #only use the x?? coordinate for the axis of rotation. Since this is 
    #not a 2d object anymore, an axis of rotation should be used instead of a point

    #Using the same y coordinates for each point, and just rotating the x and z


    #layer gen
    curve_curved=np.zeros((num_layers*points_per_layer*slices_per_layer,6))
    base_layer = np.zeros((points_per_layer,6))
    

    circle_points = PointsInCircum(tube_diameter/2, points_per_layer)

    #base layer
    print(len(circle_points)-1)
    for i in range(len(circle_points)-1):
        base_layer[i,0]=circle_points[i][0]
        base_layer[i,1]=circle_points[i][1]
        base_layer[i,-1]=-1
    

    #first layer
    for i in range(len(circle_points)-1):
        curve_curved[i,0]=circle_points[i][0]
        curve_curved[i,1]=circle_points[i][1]
        curve_curved[i,-1]=-1
        curve_curved[i,2]=vertical_shift
    # fig,ax = plt.subplots()
    # ax.plot(curve_curved[0:points_per_layer,0],curve_curved[0:points_per_layer,1],'r.-')
    # ax.set_aspect('equal')
    # ax.set_xlabel('x (mm)')
    # ax.set_ylabel('y (mm)')
    # plt.show()
    

    for layer in range(num_layers*slices_per_layer-1):
        for point in range(points_per_layer):
              #rotate x coordinates
              dx,dz = rotate([rot_point, vertical_shift], 
                             (curve_curved[layer*points_per_layer+point,0],curve_curved[layer*points_per_layer+point,2])
                             ,-layer_angle)
              
              curve_curved[(layer+1)*points_per_layer+point,0] = dx
              curve_curved[(layer+1)*points_per_layer+point,2] = dz            

              grav_dx,grav_dz = rotate((0,0), (curve_curved[layer*points_per_layer+point,3],curve_curved[layer*points_per_layer+point,5]),-layer_angle)
              curve_curved[(layer+1)*points_per_layer+point,3] = grav_dx
              curve_curved[(layer+1)*points_per_layer+point,5] = grav_dz
            
            # assign previous layer's y coordinate
              curve_curved[(layer+1)*points_per_layer+point,1] = curve_curved[layer*points_per_layer+point,1]
    vis_step=1
    #plt.rc('text', usetex=True)
    #plt.rc('font', family='serif')
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot3D(curve_curved[::vis_step,0],curve_curved[::vis_step,1],curve_curved[::vis_step,2],'r.-')
    #ax.quiver(curve_curved[::vis_step,0],curve_curved[::vis_step,1],curve_curved[::vis_step,2],curve_curved[::vis_step,3],curve_curved[::vis_step,4],curve_curved[::vis_step,5],length=10, normalize=True)
    ax.quiver(X=rot_point,Y=-20,Z=0,U=0,V=1,W=0,length = 40,color='g')

    
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_zlabel('z (mm)')
    # ax.set_xlim([-50, 250])
    # ax.set_ylim([-25, 25])
    # ax.set_zlim([0, 250])
    ax.set_aspect('equal')
    plt.locator_params(axis='y', nbins=4)

    plt.show()

    for layer in range(num_layers*slices_per_layer):
        np.savetxt(
                'slice_ER_4043_large_cold/curve_sliced/slice'+str(layer+1)+'_0.csv',
                curve_curved[layer*points_per_layer:(layer+1)*points_per_layer],delimiter=','
                )
    
    np.savetxt('slice_ER_4043_large_cold/curve_sliced/slice0_0.csv',base_layer,delimiter=',')


if __name__ == '__main__':
      main()	
