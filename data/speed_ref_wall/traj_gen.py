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
	speed = 13
	feed_speed = 160
	material = 'ER_4043'

	dH = weld_dh2v.v2dh_loglog(speed,feed_speed,material)
	
	print("dH: ", dH)

	#wall characteristics
	wall_length = 100
	points_distance=0.5
	num_layers = 31
	points_per_layer=int(wall_length/points_distance)
	vertical_shift = 3 #mm
	  
	
	#layer gen
	curve_curved=np.zeros((num_layers*points_per_layer,6))
	base_layer = np.zeros((points_per_layer,6))

	#base layer
	base_layer[0:points_per_layer,0]=np.linspace(0,wall_length,points_per_layer)
	base_layer[0:points_per_layer,2]=0
	base_layer[0:points_per_layer,-1]=-np.ones(points_per_layer)

	

	#first layer
	curve_curved[0:points_per_layer,0]=np.linspace(0,wall_length,points_per_layer)
	curve_curved[0:points_per_layer,2]=vertical_shift
	curve_curved[0:points_per_layer,-1]=-np.ones(points_per_layer)


	for layer in range(num_layers-1):
		for point in range(points_per_layer):
			  
			curve_curved[(layer+1)*points_per_layer+point,0] = curve_curved[(layer)*points_per_layer+point,0]
			curve_curved[(layer+1)*points_per_layer+point,2] = curve_curved[(layer)*points_per_layer+point,2]+dH

			  
			curve_curved[(layer+1)*points_per_layer+point,3] = curve_curved[(layer)*points_per_layer+point,3] 
			curve_curved[(layer+1)*points_per_layer+point,5] = curve_curved[(layer)*points_per_layer+point,5]


	vis_step=20
	fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
	ax.plot3D(curve_curved[::vis_step,0],curve_curved[::vis_step,1],curve_curved[::vis_step,2],'r.-')
	ax.quiver(curve_curved[::vis_step,0],curve_curved[::vis_step,1],curve_curved[::vis_step,2],curve_curved[::vis_step,3],curve_curved[::vis_step,4],curve_curved[::vis_step,5],length=10, normalize=True)
	plt.show()

	for layer in range(num_layers):
		np.savetxt('slice_ER_4043_13/curve_sliced/slice'+str(layer+1)+'_0.csv',curve_curved[layer*points_per_layer:(layer+1)*points_per_layer],delimiter=',')
	
	np.savetxt('slice_ER_4043_13/curve_sliced/slice0_0.csv',base_layer,delimiter=',')  


if __name__ == '__main__':
	main()	