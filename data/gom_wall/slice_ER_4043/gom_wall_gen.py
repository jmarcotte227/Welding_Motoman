import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
import pickle
from scipy.spatial.transform import Rotation as R
from scipy.optimize import root
from math import ceil

sys.path.append('../../weld/')
from weld_gom_height import v2dh_loglog, dh2v_loglog
from gom_funcs import distance_out
from gom_funcs import v_rat_out

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


     
########################################################################
# limits of welding bead height (based on min and Max estimate speed from Eric)
    # needs to be modified based on actual limits from Eric 
def main():
    min_speed = 7.0
    max_speed = 7.001
    feed_speed = 230
    material = 'ER_4043'

    max_dH = v2dh_loglog(min_speed,feed_speed,material)
    min_dH = v2dh_loglog(max_speed,feed_speed,material)
    mean_dH = (max_dH+min_dH)/2

    print('Max dH: ', max_dH)
    print('Min dH: ', min_dH)
    print('Mean dH: ', mean_dH)
    print('------------------------')

    #wall characteristics
    wall_length = 200 
    wall_width = 40
    points_distance=0.5
    num_layers = 10
    vertical_shift = 0 #mm

    #segment characteristics
    seg_len = wall_width #mm
    points_distance=0.5
    points_per_segment=int(seg_len/points_distance)
    

    #base_seg params
    num_pre_seg = 2
    base_seg_len = 100
    base_seg_min = 0
    base_seg_max = 10
    work_offset = 30
    base_start_offset = 50
    points_per_base_seg = int(base_seg_len/points_distance)
    base_seg_curve = np.zeros((num_pre_seg*points_per_base_seg,6))

    #first baseseg
    base_seg_curve[0:points_per_base_seg, 0] = np.linspace(base_start_offset, base_start_offset+base_seg_len, points_per_base_seg)
    base_seg_curve[0:points_per_base_seg, 1] = base_seg_min
    base_seg_curve[0:points_per_base_seg, -1] = -1

    #second baseseg
    base_seg_curve[points_per_base_seg:points_per_base_seg*2, 0] = np.linspace(base_start_offset, base_start_offset+base_seg_len, points_per_base_seg)
    base_seg_curve[points_per_base_seg:points_per_base_seg*2, 1] = base_seg_max
    base_seg_curve[points_per_base_seg:points_per_base_seg*2, -1] = -1

   

    #rotation criteria
    layer_angle = np.arcsin((max_dH-min_dH)/wall_length)
    rot_point = max_dH/np.tan(layer_angle)
    print('Layer Angle:', np.rad2deg(layer_angle))
    print('Final Angle:', np.rad2deg(layer_angle)*num_layers)
    print('Point of Rotation:', rot_point)
      
    


    #layer gen
    # curve_curved=np.zeros((num_layers**points_per_layer,6))
    # base_layer = np.zeros((points_per_layer,6))

    
    #plotting baselayer
    vis_step=1
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot3D(base_seg_curve[::vis_step,0],base_seg_curve[::vis_step,1],base_seg_curve[::vis_step,2],'b.-')
    #ax.quiver(curve_curved[::vis_step,0],curve_curved[::vis_step,1],curve_curved[::vis_step,2],curve_curved[::vis_step,3],curve_curved[::vis_step,4],curve_curved[::vis_step,5],length=10, normalize=True)
    ax.set_aspect('equal')
    plt.show()
    #save as individual segments
    for seg_idx in range(num_pre_seg):
        segment = base_seg_curve[points_per_base_seg*(seg_idx):points_per_base_seg*(seg_idx+1), :]
        np.savetxt('slice_ER_4043/curve_sliced/slice0_'+str(seg_idx)+'.csv',segment,delimiter=',')


    ############# first layer ##################
    # Layer gen functions
    def prof(height_des, h_start, h_end, wall_length, dOffset):
        return(height_des-h_start)*wall_length/(h_end-h_start)-dOffset

    def distance_func(v_t, v_w, v_t_prev, v_w_prev, v_t_fill, v_w_fill, hProfStart, hProfEnd, wall_length, dOffset):
        h = v2dh_loglog(v_t, v_w)
        hStart = v2dh_loglog(v_t_prev, v_w_prev)
        vRatStart = (v_w_prev/39.37)/(v_t_prev*0.06)
        vRatCurr = (v_w/39.37)/(v_t*0.06)
        vRatFill = (v_w_fill/39.37)/(v_t_fill*0.06)
        d_fill = distance_out(vRatStart, hStart, vRatCurr, h, vRatFill)
        d_prof = prof(h, hProfStart, hProfEnd, wall_length, dOffset)
        return d_prof-d_fill
    
    vt_prev = max_speed
    vw_prev = 230
    vw_next = 230
    vt_fill = 5
    vw_fill = 280

    #initialized with first bead params
    print(min_dH)
    print(vt_prev)
    distances = [0.0]
    heights = [min_dH]
    vel_profile = [vt_prev]
    cum_distance = 0
    
    

    while(cum_distance<wall_length):
        sol = root(distance_func, 1, (vw_next, vt_prev, vw_prev, vt_fill, vw_fill, min_dH, max_dH, wall_length, cum_distance))
        vel_profile.append(sol.x[0])
        sol_h = v2dh_loglog(sol.x, 230)
        sol_h = sol_h[0]
        sol_d = prof(sol_h, min_dH, max_dH, wall_length, cum_distance)
        heights.append(sol_h)
        distances.append(sol_d+cum_distance)
        cum_distance = cum_distance+sol_d
        vt_prev = sol.x
    print(distances)
    print(heights)
    fig,ax = plt.subplots(1,1)
    ax.plot(np.linspace(min_dH, max_dH, 100))
    ax.scatter(distances, heights)
    plt.show()
    print("Num matrix segs: ", len(distances))
    print(vel_profile)
    num_matrix_seg = len(distances)
    num_filler_seg = num_matrix_seg - 1
    points_per_layer = num_matrix_seg*points_per_segment+num_filler_seg*points_per_segment # update once connecting segments are interpolated
    curve_curved=np.zeros((num_layers*points_per_layer,6))

    ######## Create the first layer ###############

    # matrix beads

    for seg_idx in range(num_matrix_seg):
        curve_curved[seg_idx*points_per_segment:(seg_idx+1)*points_per_segment, 0] = distances[seg_idx]
        curve_curved[seg_idx*points_per_segment:(seg_idx+1)*points_per_segment, 1] = np.linspace(work_offset, work_offset+wall_width, points_per_segment)
        curve_curved[seg_idx*points_per_segment:(seg_idx+1)*points_per_segment, -1] = -1
    # filler beads
    for seg_idx in range(num_filler_seg):
        curve_curved[(num_matrix_seg+seg_idx)*points_per_segment:(num_matrix_seg+seg_idx+1)*points_per_segment, 0] = distances[seg_idx]+(distances[seg_idx+1]-distances[seg_idx])/2
        curve_curved[(num_matrix_seg+seg_idx)*points_per_segment:(num_matrix_seg+seg_idx+1)*points_per_segment, 1] = np.linspace( work_offset+wall_width, work_offset, points_per_segment)
        curve_curved[(num_matrix_seg+seg_idx)*points_per_segment:(num_matrix_seg+seg_idx+1)*points_per_segment, -1] = -1
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot3D(base_seg_curve[::vis_step,0],base_seg_curve[::vis_step,1],base_seg_curve[::vis_step,2],'b.-')
    ax.plot3D(curve_curved[::vis_step,0],curve_curved[::vis_step,1],curve_curved[::vis_step,2],'r.-')
    
    #ax.quiver(curve_curved[::vis_step,0],curve_curved[::vis_step,1],curve_curved[::vis_step,2],curve_curved[::vis_step,3],curve_curved[::vis_step,4],curve_curved[::vis_step,5],length=10, normalize=True)
    ax.set_aspect('equal')
    plt.show()
    dir_flag = True
    for layer in range(num_layers-1):
        for point in range(points_per_layer):
              dx,dz = rotate([wall_length-rot_point, 0], (curve_curved[layer*points_per_layer+point,0],curve_curved[layer*points_per_layer+point,2]),layer_angle)
              #print(dx,dz)
              curve_curved[(layer+1)*points_per_layer+point,0] = dx
              if dir_flag:
                curve_curved[(layer+1)*points_per_layer+point,1] = -curve_curved[point,1] + 2*work_offset+wall_width
              elif not dir_flag:
                curve_curved[(layer+1)*points_per_layer+point,1] = curve_curved[point,1]
              curve_curved[(layer+1)*points_per_layer+point,2] = dz

              grav_dx,grav_dz = rotate((0,0), (curve_curved[layer*points_per_layer+point,3],curve_curved[layer*points_per_layer+point,5]),layer_angle)
              curve_curved[(layer+1)*points_per_layer+point,3] = grav_dx
              curve_curved[(layer+1)*points_per_layer+point,5] = grav_dz

        dir_flag = not dir_flag

    #export velocity profile
    vel_profile_export = np.zeros(num_matrix_seg+num_filler_seg)
    vel_profile_export[:num_matrix_seg] = vel_profile
    vel_profile_export[num_matrix_seg:] = vt_fill
    feed_profile = np.zeros(num_matrix_seg+num_filler_seg)
    for seg in range(num_matrix_seg):
        feed_profile[seg] = vw_prev
    for seg in range(num_matrix_seg, num_matrix_seg+num_filler_seg):
        feed_profile[seg] = vw_fill
    
    print(vel_profile_export)
    print(feed_profile)
    with open('slice_ER_4043/vel_profile.pkl', 'wb') as file:
        pickle.dump(vel_profile_export, file)
    with open('slice_ER_4043/feed_profile.pkl', 'wb') as file:
        pickle.dump(feed_profile, file)
    vis_step=1
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot3D(base_seg_curve[::vis_step,0],base_seg_curve[::vis_step,1],base_seg_curve[::vis_step,2],'b.-')
    ax.plot3D(curve_curved[::vis_step,0],curve_curved[::vis_step,1],curve_curved[::vis_step,2],'r.-')
    
    #ax.quiver(curve_curved[::vis_step,0],curve_curved[::vis_step,1],curve_curved[::vis_step,2],curve_curved[::vis_step,3],curve_curved[::vis_step,4],curve_curved[::vis_step,5],length=10, normalize=True)
    ax.set_aspect('equal')
    plt.show()

    for layer in range(num_layers):
         for seg in range(num_matrix_seg+num_filler_seg):
	         np.savetxt('slice_ER_4043/curve_sliced/slice'+str(layer+1)+'_'+str(seg)+'.csv',
                     curve_curved[layer*points_per_layer+seg*points_per_segment:layer*points_per_layer+(1+seg)*points_per_segment],delimiter=',')
        


if __name__ == '__main__':
    main()	