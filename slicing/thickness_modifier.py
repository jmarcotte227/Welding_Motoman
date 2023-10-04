import numpy as np
import glob


#Tunable Parameters
width_of_bead = 10 #mm
wavelength = 5 #mm

for file in glob.glob('curve_sliced/*.csv'): 
	curve=np.loadtxt(file,delimiter=',')
	
    #define start point to right of inital point by width_of_bead/2

    #proceed along trajectory by 1/2 wavelength

    #calculate point perpendicular to trajectory to the left by width_of_bead/2

    #loop, incrementing between points