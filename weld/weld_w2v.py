"""Module containing the data for the width model for WAAM"""

import numpy as np

material_param = {}
material_param['ER_4043'] = {
    #ER 4043
    #update with better estimate once found error: -0.009853848161215595
    "160ipm": [-0.55863074,  3.01451518], 
    "200ipm": [-0.33208935,  2.72353279]
}

def w2v_loglog(w, mode = 160, material='ER_4043'):
    mode=int(mode)
    param = material_param[material][str(mode)+'ipm']

    logw = np.log(w)
    logv = (logw-param[1])/param[0]
    
    v = np.exp(logv)
    return v

def v2w_loglog(v, mode = 160, material='ER_4043'):
    mode=int(mode)
    #print(str(mode)+'ipm')
    param = material_param[material][str(mode)+'ipm']
    logw = param[0]*np.log(v)+param[1]
    
    w = np.exp(logw)
    return w

def main():
    #verification values
    ver_vals = [13.75]
    vels = v2w_loglog(ver_vals,200)
    print("velocities: ", vels)

if __name__=="__main__":
    main()
