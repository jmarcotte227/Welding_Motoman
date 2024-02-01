import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
from scipy.spatial.transform import Rotation as R

sys.path.append('../../weld/')
from weld_dh2v import v2dh_loglog, dh2v_loglog
from weld_w2v import w2v_loglog, v2w_loglog
from overlap_distance import overlap_distance

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

def height_profile_func(minh, maxh, wall_length, x):
     return ((maxh-minh)/float(wall_length))*x+minh

def bead_profile(h, w, x, d):
     return (-4*h/w**2)*(x-d)**2+h
     
########################################################################
# limits of welding bead height (based on min and Max estimate speed from Eric)
    # needs to be modified based on actual limits from Eric 
def main():
    min_speed = 5
    max_speed = 15
    feed_speed = 160
    material = 'ER_4043'

    max_dH = v2dh_loglog(min_speed,feed_speed,material)
    min_dH = v2dh_loglog(max_speed,feed_speed,material)
    mean_dH = (max_dH+min_dH)/2

    print('Max dH: ', max_dH)
    print('Min dH: ', min_dH)
    print('Mean dH: ', mean_dH)
    print('------------------------')

    #wall characteristics
    wall_length = 100 
    wall_width = 30
    points_distance=0.5
    num_layers = 31
    points_per_segment=int(wall_width/points_distance)
    vertical_shift = 3 #mm

    #other parameters
    prof_increment = 0.1
    vel_increment = 0.05

    #rotation criteria
    layer_angle = np.arcsin((max_dH-min_dH)/wall_length)
    rot_point = max_dH/np.tan(layer_angle)
    print('Layer Angle:', np.rad2deg(layer_angle))
    print('Final Angle:', np.rad2deg(layer_angle)*num_layers)
    print('Point of Rotation:', rot_point)
      
    


    #layer gen
    # curve_curved=np.zeros((num_layers**points_per_layer,6))
    # base_layer = np.zeros((points_per_layer,6))

    #base layer
    # base_layer[0:points_per_layer,0]=np.linspace(0,wall_length,points_per_layer)
    # base_layer[0:points_per_layer,2]=0
    # base_layer[0:points_per_layer,-1]=-np.ones(points_per_layer)

    #np.savetxt('slice_ER_4043/curve_sliced/slice0_0.csv',base_layer,delimiter=',')

    ############# first layer ##################
    initial_layer=np.zeros((points_per_segment*100,6))

    fig_sim, ax_sim = plt.subplots(1,1)
    ax_sim.set_xlim(-5, 105)
    ax_sim.set_ylim(0, 3.5)
    ax_sim.set_aspect('equal')
    x_sim = np.linspace(-5,wall_length+5, 1010)
    #setting first known element of vel profile
    vel_profile = [dh2v_loglog(min_dH, feed_speed, material)]
    height_profile = [min_dH]
    distance_of_height = [0]
    #first segment\
    initial_layer[0:points_per_segment,0]= np.ones(points_per_segment) * wall_length
    initial_layer[0:points_per_segment,1]=np.linspace(0,wall_width,points_per_segment)
    initial_layer[0:points_per_segment,2]=vertical_shift
    initial_layer[0:points_per_segment,-1]=-np.ones(points_per_segment)

    #find offset distance
    h_curr = v2dh_loglog(vel_profile[0], feed_speed, material)
    w_curr = v2w_loglog(vel_profile[0], feed_speed, material)

    next_vel = vel_profile[0]-vel_increment
    
    h_next = v2dh_loglog(next_vel, feed_speed, material)
    w_next = v2w_loglog(next_vel, feed_speed, material)

    distance, a, b, x_1, x_2, y_1, y_2 = overlap_distance(h_curr, w_curr, h_next, w_next)
    prof_next_height = height_profile_func(min_dH, max_dH, wall_length, distance)

    #prelim plot
    ax_sim.plot(x_sim, bead_profile(h_curr, w_curr, x_sim, 0), color='b')
    idx = 1
    cum_distance = 0
    dir_flag = True


    while(cum_distance<wall_length):
        while(prof_next_height>h_next):
            next_vel = next_vel-vel_increment
            h_next = v2dh_loglog(next_vel, feed_speed, material)
            w_next = v2w_loglog(next_vel, feed_speed, material)

            distance, a, b, x_1, x_2, y_1, y_2 = overlap_distance(h_curr, w_curr, h_next, w_next)
            print('distance:' , distance)
            prof_next_height = height_profile_func(min_dH, max_dH, wall_length, cum_distance+distance)

        
        ax_sim.plot([x_1+cum_distance, x_2+cum_distance], [y_1, y_2], color='b')

        cum_distance = cum_distance+distance

        ax_sim.plot(x_sim, bead_profile(h_next, w_next, x_sim, cum_distance), color = 'b')
        vel_profile.append(next_vel)
        height_profile.append(h_next)
        distance_of_height.append(cum_distance)
        
        print("Height Profile Height: ", prof_next_height)
        print("Bead height: ", h_next)
        initial_layer[idx*points_per_segment:(idx+1)*points_per_segment,0]=wall_length-cum_distance
        if dir_flag:
            initial_layer[idx*points_per_segment:(idx+1)*points_per_segment,1]=np.linspace(wall_width, 0,points_per_segment)
        else:
            initial_layer[idx*points_per_segment:(idx+1)*points_per_segment,1]=np.linspace(0, wall_width,points_per_segment)
        initial_layer[idx*points_per_segment:(idx+1)*points_per_segment,2]=vertical_shift
        initial_layer[idx*points_per_segment:(idx+1)*points_per_segment,-1]=-np.ones(points_per_segment)

        h_curr = h_next
        w_curr = w_next
        
        

        next_vel = next_vel-vel_increment

        h_next = v2dh_loglog(next_vel, feed_speed, material)
        w_next = v2w_loglog(next_vel, feed_speed, material)

        distance, a, b, x_1, x_2, y_1, y_2 = overlap_distance(h_curr, w_curr, h_next, w_next)
        prof_next_height = height_profile_func(min_dH, max_dH, wall_length, cum_distance+distance)

        dir_flag = not dir_flag
        idx = idx+1

    #plt.show()
    
    
    fig, ax = plt.subplots(1,1)
    x = np.linspace(0, wall_length, 100)
    ax.plot(distance_of_height,height_profile)
    ax.plot(x, height_profile_func(min_dH, max_dH, wall_length, x))
    #plt.show()
    
    #eliminate empty elements of list
    idx = 0
    while(initial_layer[idx*points_per_segment,-1]):
         idx+=1
    end_idx = idx
    initial_layer = initial_layer[:(end_idx)*points_per_segment, :]
    
    points_per_layer = end_idx*points_per_segment # update once connecting segments are interpolated
    curve_curved=np.zeros((num_layers*points_per_layer,6))
    curve_curved[:points_per_layer, :] = initial_layer[:,:]
    for layer in range(num_layers-1):
        for point in range(points_per_layer):
              dx,dz = rotate([rot_point, 0], (curve_curved[layer*points_per_layer+point,0],curve_curved[layer*points_per_layer+point,2]),-layer_angle)
              #print(dx,dz)
              curve_curved[(layer+1)*points_per_layer+point,0] = dx
              curve_curved[(layer+1)*points_per_layer+point,1] = curve_curved[point,1]
              curve_curved[(layer+1)*points_per_layer+point,2] = dz

              grav_dx,grav_dz = rotate((0,0), (curve_curved[layer*points_per_layer+point,3],curve_curved[layer*points_per_layer+point,5]),-layer_angle)
              curve_curved[(layer+1)*points_per_layer+point,3] = grav_dx
              curve_curved[(layer+1)*points_per_layer+point,5] = grav_dz


    vis_step=1
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot3D(curve_curved[::vis_step,0],curve_curved[::vis_step,1],curve_curved[::vis_step,2],'r.-')
    #ax.quiver(curve_curved[::vis_step,0],curve_curved[::vis_step,1],curve_curved[::vis_step,2],curve_curved[::vis_step,3],curve_curved[::vis_step,4],curve_curved[::vis_step,5],length=10, normalize=True)
    ax.set_aspect('equal')
    plt.show()

    for layer in range(num_layers):
	    np.savetxt('slice_ER_4043/curve_sliced/slice'+str(layer+1)+'_0.csv',curve_curved[layer*points_per_layer:(layer+1)*points_per_layer],delimiter=',')
        


if __name__ == '__main__':
      main()	