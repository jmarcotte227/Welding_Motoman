import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root

from weld_gom_height import v2dh_loglog, dh2v_loglog


def distance_out(v_rat_1, h1, v_rat_2, h2, v_rat_fill, rad_wire=1.2/2):
    d = (np.pi*rad_wire**2)*(2*v_rat_fill+v_rat_1+v_rat_2)/(h1+h2)
    return d

def v_rat_out(v_rat_1, h1, v_rat_2, h2, d, rad_wire=1.2/2):
    v_rat = ((d*(h1+h2))/(2*np.pi*rad_wire**2))-((v_rat_1+v_rat_2)/2)
    return v_rat

if __name__=='__main__':

    def prof(height_des, h_start, h_end, wall_length, dOffset):
        return(height_des-h_start)*wall_length/(h_end-h_start)-dOffset

    def distance_func(v_t, v_w, v_t_prev, v_w_prev, v_t_fill, v_w_fill, hProfStart, hProfEnd, wall_length, dOffset):
        print("VT: ", v_t)
        h = v2dh_loglog(v_t, v_w)
        hStart = v2dh_loglog(v_t_prev, v_w_prev)
        vRatStart = (v_w_prev/39.37)/(v_t_prev*0.06)
        vRatCurr = (v_w/39.37)/(v_t*0.06)
        vRatFill = (v_w_fill/39.37)/(v_t_fill*0.06)
        d_fill = distance_out(vRatStart, hStart, vRatCurr, h, vRatFill)
        d_prof = prof(h, hProfStart, hProfEnd, wall_length, dOffset)
        return d_prof-d_fill

    feed_conv = 1/39.37
    vel_conv = 0.06
    vw_start = 230        # ipm    
    vt_start = 12       # mm/s     
    v_start_rat = (vw_start*feed_conv)/(vt_start*vel_conv)
    h_start = v2dh_loglog(vt_start, vw_start) # mm
    print("H start: ", h_start)
    h_end = v2dh_loglog(4, 230)


    vwf = 240*feed_conv
    vwt = 6.08*vel_conv
    v_rat_fill = vwf/vwt

    d_start = 6.6     # mm
    rad = 1.2/2       # mm
    d_offset = 80


    
    vw2 = 230         #ipm
    dist_1 = np.zeros(100)
    height_1 = np.zeros(100)
    for idx,vt_2 in enumerate(np.linspace(12.0,4.0,100)):
        h_2 = v2dh_loglog(vt_2, vw2)
        v_2_rat = (vw2*feed_conv)/(vt_2*vel_conv)
        height_1[idx] = h_2
        dist_1[idx]=distance_out(v_start_rat, h_start, v_2_rat, h_2, v_rat_fill)

    # vw2 = 200         #ipm
    # dist_2 = np.zeros(100)
    # height_2 = np.zeros(100)
    # for idx,vt_2 in enumerate(np.linspace(7,5,100)):
    #     h_2 = v2dh_loglog(vt_2, vw2)
    #     print("in loop")
    #     v_2_rat = (vw2*feed_conv)/(vt_2*vel_conv)
    #     height_2[idx] = h_2
    #     dist_2[idx]=distance_out(v_start_rat, h_start, v_2_rat, h_2, v_rat_fill)

    # fig,ax = plt.subplots(1,2)
    # ax[0].plot(np.linspace(12,4,100), height_1)
    # ax[1].plot(np.linspace(12,4,100), dist_1)
    # ax[0].plot(np.linspace(7,5,100), height_2)
    # ax[1].plot(np.linspace(7,5,100), dist_2)

    sol = root(distance_func, 1, (vw2, vt_start, vw_start, 6.08, 240, h_start, h_end, 100, d_offset))
    print(sol)

    sol_h = v2dh_loglog(sol.x, 230)
    sol_d = prof(sol_h, h_start, h_end,100,d_offset)
    print(sol_h)
    print(sol_d)
    fig,ax = plt.subplots(1,1)
    ax.plot(np.linspace(h_start, h_end, 100))
    ax.plot(dist_1+d_offset,height_1)
    ax.scatter(sol_d+d_offset, sol_h)
    plt.show()


