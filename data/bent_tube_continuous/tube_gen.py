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

def PointsInCircum(r,n):
    return [(np.cos(2*np.pi/n*x)*r,np.sin(2*np.pi/n*x)*r) for x in range(0,n+1)]
########################################################################
# limits of welding bead height (based on min and Max estimate speed from Eric)
    # needs to be modified based on actual limits from Eric 
def main():
    min_speed = 5
    max_speed = 15
    feed_speed = 160
    material = 'ER_4043'

    max_dH = weld_dh2v.v2dh_loglog(min_speed,feed_speed,material)*10
    min_dH = weld_dh2v.v2dh_loglog(max_speed,feed_speed,material)*10
    mean_dH = (max_dH+min_dH)/2

    print('Max dH: ', max_dH)
    print('Min dH: ', min_dH)
    print('Mean dH: ', mean_dH)
    print('------------------------')

    #tube characteristics
    tube_diameter = 50
    num_layers = 5
    points_per_layer=50
    point_distance = np.pi*tube_diameter/points_per_layer
    vertical_shift = 3 #mm

    print('Point distance: ', point_distance)
    print('------------------------')


    #rotation criteria
    layer_angle = np.arcsin((max_dH-min_dH)/tube_diameter)
    rot_point = max_dH/np.tan(layer_angle)-tube_diameter/2
    
    print('Layer Angle:', np.rad2deg(layer_angle))
    print('Final Angle:', np.rad2deg(layer_angle)*num_layers)
    print('Point of Rotation:', rot_point)
      
    #only use the x?? coordinate for the axis of rotation. Since this is 
    #not a 2d object anymore, an axis of rotation should be used instead of a point

    #Using the same y coordinates for each point, and just rotating the x and z


    #layer gen
    curve_curved=np.zeros((num_layers*points_per_layer,6))
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

    ############# bend the trajectory by rotating and scaling
    center_point_idx = int(points_per_layer/2)
    center_point = circle_points[center_point_idx]

    # calculate angle from outer point
    dx_inner, dz_inner = rotate([rot_point, vertical_shift],
                                (curve_curved[0,0], curve_curved[0,2]), -layer_angle/2)
    adj = dx_inner - curve_curved[center_point_idx, 0]
    hyp = np.sqrt(adj**2 + (dz_inner-vertical_shift)**2)
    seg_rot_angle = np.arccos(adj/hyp)

    # calculate scale factor
    scale_factor = hyp/tube_diameter

    # rotate downhill segment to match
    for point in range(0, center_point_idx):
        dx, dz = rotate([curve_curved[center_point_idx, 0], curve_curved[center_point_idx, 2]], 
                                  (curve_curved[point,0], curve_curved[point,2]),
                                  -seg_rot_angle)
        
        curve_curved[point,0] = (dx-curve_curved[center_point_idx,0])*scale_factor + curve_curved[center_point_idx,0]
        curve_curved[point,2] = (dz-curve_curved[center_point_idx,2])*scale_factor + curve_curved[center_point_idx,2]

        grav_dx,grav_dz = rotate((0,0), (curve_curved[point,3],curve_curved[point,5]),layer_angle/points_per_layer*(center_point_idx-point))
        curve_curved[point,3] = grav_dx
        curve_curved[point,5] = grav_dz

    # rotate uphill segment
    for point in range(center_point_idx, points_per_layer):
        dx, dz = rotate([curve_curved[center_point_idx, 0], curve_curved[center_point_idx, 2]], 
                                  (curve_curved[point,0], curve_curved[point,2]),
                                  seg_rot_angle)
        curve_curved[point,0] = (dx-curve_curved[center_point_idx,0])*scale_factor + curve_curved[center_point_idx,0]
        curve_curved[point,2] = (dz-curve_curved[center_point_idx,2])*scale_factor + curve_curved[center_point_idx,2]
        grav_dx,grav_dz = rotate((0,0), (curve_curved[point,3],curve_curved[point,5]),layer_angle/points_per_layer*(center_point_idx-point))
        curve_curved[point,3] = grav_dx
        curve_curved[point,5] = grav_dz

    print("center point: ", center_point)
    print("dx_inner: ", dx_inner)
    print("dz_inner: ", dz_inner)
    print("adjacent: ", adj)
    print("hypotenuse: ", hyp)
    print("new angle: ", seg_rot_angle)

    #rotate 
    

    fig,ax = plt.subplots()
    ax.plot(curve_curved[0:points_per_layer,0],curve_curved[0:points_per_layer,1],'r.-')
    ax.plot(dx_inner, dz_inner)
    ax.set_aspect('equal')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    plt.show()
    
    

    for layer in range(num_layers-1):
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
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot3D(curve_curved[::vis_step,0],curve_curved[::vis_step,1],curve_curved[::vis_step,2],'r.-')
    #ax.plot3D(base_layer[::vis_step,0],base_layer[::vis_step,1],base_layer[::vis_step,2],'b.-')
    #ax.quiver(curve_curved[::vis_step,0],curve_curved[::vis_step,1],curve_curved[::vis_step,2],curve_curved[::vis_step,3],curve_curved[::vis_step,4],curve_curved[::vis_step,5],length=10, normalize=True)
    #ax.quiver(X=rot_point,Y=-20,Z=0,U=0,V=1,W=0,length = 40,color='g')
    print()
    ax.set_aspect('equal')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_zlabel('z (mm)')
    plt.locator_params(axis='y', nbins=4)

    plt.show()

    ######### uncomment for export ####################
    # np.savetxt('slice_ER_4043/curve_sliced/slice1_0.csv',curve_curved,delimiter=',') 
    # np.savetxt('slice_ER_4043/curve_sliced/slice0_0.csv',base_layer,delimiter=',')


if __name__ == '__main__':
      main()	
