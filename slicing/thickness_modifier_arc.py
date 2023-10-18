import numpy as np
import glob
import math
import matplotlib.pyplot as plt
import os

#ONLY WORKS FOR WALL
#Tunable Parameters
width_of_bead = 10 #mm
wavelength = 10 #mm

int_distance = 0.1 #mm distance between interpolated points along curve 

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
        #define start point to the left of the wall
        start_point = np.add(curve[0],np.array([0,width_of_bead/2,0,0,0,0]))
        thick_curve.append(start_point)
        #prev_point is the point that was just logged along the trajectory
        #next_point is the point next along the trajectory at a fixed interval
        i = 1
        peak_dir = False # True is right, False is left
        for point in curve:
            try:
                curve_prev_peak = curve[i-1]
                curve_next_peak = curve[i]
                next_point_dist = magnitude(curve_prev_peak[0:2]-curve_next_peak[0:2])
                #find next point that is 1/2 wavelength apart from previous point
            
                while(next_point_dist < wavelength/2): 
                    i += 1
                    curve_next_peak = curve[i]
                    component_dif = curve_next_peak-curve_prev_peak
                    next_point_dist = magnitude(component_dif[0:2])
                if peak_dir == False:
                    next_peak = np.add(curve_next_peak, [0,-width_of_bead/2,0,0,0,0] )
                elif peak_dir == True:
                    next_peak = np.add(curve_next_peak, [0,width_of_bead/2,0,0,0,0] )
                peak_dir = not peak_dir
                
                #calculate arc parameters
                v_curve2peak = np.subtract(next_peak[0:3],curve_next_peak[0:3]) #vector from curve to peak
                radius_point = np.cross(next_peak[3:6],v_curve2peak)
                if peak_dir == False:
                    radius_point = -1*radius_point #negates the direction of the crossproduct for one side

                
                print(next_peak[3:6])
                print(v_curve2peak)
                print(radius_point)
                exit()

                #thick_curve.append(peak_point)
                thick_curve.append(next_peak)
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