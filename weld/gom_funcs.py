import numpy as np

def distance_out(v_rat_1, h1, v_rat_2, h2, v_rat_fill, rad_wire):
    d = (np.pi*rad_wire**2)*(2*v_rat_fill+v_rat_1+v_rat_2)/(h1+h2)
    return d

def v_rat_out(v_rat_1, h1, v_rat_2, h2, d, rad_wire):
    
    v_rat = ((h1+h2)/(2*np.pi*rad_wire**2))*d-(v_rat_1+v_rat_2)/2
    return v_rat

if __name__=='__main__':
    vw1 = 5         # m/min     200 ipm
    vt1 = 0.4       # m/min     6.6 mm/s ish
    h_1 = 0.0025    # m
    
    vw2 = 6         # m/min     240 ipm
    vt2 = 0.7       # m/min     11.6 mm/s ish
    h_2 = 0.00102      # m

    vwf = 6
    vwt = 0.36

    d_start = 0.005     # m
    rad = 0.0012       # m
    

    dist = distance_out(vw1/vt1, h_1, vw2/vt2, h_2, vwf/vwt, rad)
    vf_rat = v_rat_out(vw1/vt1, h_1, vw2/vt2, h_2, dist, rad)
    print(dist)
    print(vf_rat)


