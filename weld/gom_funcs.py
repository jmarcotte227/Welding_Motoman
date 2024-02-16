import numpy as np

def distance_out(v_rat_1, h1, v_rat_2, h2, v_rat_fill, rad_wire=1.2/2):
    d = (np.pi*rad_wire**2)*(2*v_rat_fill+v_rat_1+v_rat_2)/(h1+h2)
    return d

def v_rat_out(v_rat_1, h1, v_rat_2, h2, d, rad_wire=1.2/2):
    v_rat = ((d*(h1+h2))/(2*np.pi*rad_wire**2))-((v_rat_1+v_rat_2)/2)
    return v_rat

if __name__=='__main__':
    vw2 = 5         # m/min     200 ipm
    vt2 = 0.4       # m/min     6.6 mm/s ish
    h_2 = 4       # mm
    
    vw1 = 6         # m/min     240 ipm
    vt1 = 0.7       # m/min     11.6 mm/s ish
    h_1 = 3  # mm

    vwf = 6
    vwt = 0.36

    d_start = 6.6     # mm
    rad = 1.2/2       # mm


    print("area of bead 1: ", np.pi*rad**2*vw1/vt1)
    print("area of bead 2: ", np.pi*rad**2*vw2/vt2)
    print("Aread of trapezoid: ", (h_1+h_2)/2*d_start)
    dist = distance_out(vw1/vt1, h_1, vw2/vt2, h_2, vwf/vwt, rad)
    vf_rat = v_rat_out(vw1/vt1, h_1, vw1/vt1, h_1, d_start, rad)
    print(dist)
    print(vf_rat)


