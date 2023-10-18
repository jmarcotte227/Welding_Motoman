import numpy as np
import glob
import math
import matplotlib.pyplot as plt
import os

#ONLY WORKS FOR WALL
#Tunable Parameters
width_of_bead = 10 #mm
wavelength = 10 #mm

def magnitude(vector): 
    return math.sqrt(sum(pow(element, 2) for element in vector))

if __name__ == "__main__":
    os.makedirs('curve_sliced_thick', exist_ok=True)
    #initialize plot
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.axes.set_xlim3d(left=-50, right=50) 
    ax.axes.set_ylim3d(bottom=-10, top=10) 
    ax.axes.set_zlim3d(bottom=0, top=100)

    for file in glob.glob('curve_sliced_relative/*.csv'): 
        curve=np.loadtxt(file,delimiter=',')
        thick_curve = []
        #define start point at end of wall
        thick_curve.append(curve[0])
        #prev_point is the point that was just logged along the trajectory
        #next_point is the point next along the trajectory at a fixed interval
        i = 1
        peak_dir = True # True is right, False is left
        for point in curve:
            try:
                prev_point = curve[i-1]
                next_point = curve[i]
                next_point_dist = magnitude(prev_point[0:2]-next_point[0:2])
                #find next point that is 1/2 wavelength apart from previous point
            
                while(next_point_dist < wavelength/2): 
                    i += 1
                    next_point = curve[i]
                    component_dif = next_point-prev_point
                    next_point_dist = magnitude(component_dif[0:2])
                if peak_dir == True: peak_point = [prev_point[0] + component_dif[0]/2, width_of_bead/2, 
                                                    prev_point[2], prev_point[3], prev_point[4], prev_point[5]]
                if peak_dir == False: peak_point = [prev_point[0] + component_dif[0]/2, width_of_bead/-2, 
                                                    prev_point[2], prev_point[3], prev_point[4], prev_point[5]]
                peak_dir = not peak_dir
                #append peak_point but to the side and at same z as both
                thick_curve.append(peak_point)
                thick_curve.append(next_point)
            except IndexError as e:
                print('End of layer')
                
        np_thick_curve = np.asarray(thick_curve, dtype=np.float32)
        print(file)
        #np.savetxt('curve_sliced_thick/'+os.path.basename(file),np_thick_curve,delimiter=',')
        
        vis_step=1
        
        ax.plot3D(np_thick_curve[::vis_step,0],np_thick_curve[::vis_step,1],np_thick_curve[::vis_step,2],'r.-')
        #ax.quiver(np_thick_curve[::vis_step,0],np_thick_curve[::vis_step,1],np_thick_curve[::vis_step,2],np_thick_curve[::vis_step,3],np_thick_curve[::vis_step,4],np_thick_curve[::vis_step,5],length=1, normalize=True)
        break
    plt.show()