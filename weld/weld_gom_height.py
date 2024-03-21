import numpy as np

material_param = {}
material_param['ER_4043'] = {
    #ER 4043
    "200ipm": [-0.58853786,  2.36925842],
    "210ipm": [-0.57756141,  2.43079976],
    "220ipm": [-0.71905278,  2.67732977],
    "230ipm": [-0.59095092,  2.47071631],
    "240ipm": [-0.6856249,   2.19719422],
}


def v2dh_loglog(v,mode=140,material='ER_4043'):
    
    mode=int(mode)
    #print(str(mode)+'ipm')
    param = material_param[material][str(mode)+'ipm']
    logdh = param[0]*np.log(v)+param[1]
    
    dh = np.exp(logdh)
    return dh

def dh2v_loglog(dh,mode=140,material='ER_4043'):
    
    mode=int(mode)
    param = material_param[material][str(mode)+'ipm']

    logdh = np.log(dh)
    logv = (logdh-param[1])/param[0]
    
    v = np.exp(logv)
    return v

def dh2v_quadratic(dh,mode=140):

    if mode==140:
        # 140 ipm
        a=0.006477
        b=-0.2362
        c=3.339-dh
    elif mode==160:
        # 160 ipm
        a=0.006043
        b=-0.2234
        c=3.335-dh
    
    v=(-b-np.sqrt(b**2-4*a*c))/(2*a)
    return v

def v2dh_quadratic(v,mode=140):

    if mode==140:
        # 140 ipm
        a=0.006477
        b=-0.2362
        c=3.339
    elif mode==100:
        # 100ipm
        a=0.01824187
        b=-0.58723623
        c=5.68282353
    elif mode==160:
        # 160 ipm
        a=0.006043
        b=-0.2234
        c=3.335
    
    dh = a*(v**2)+b*v+c
    return dh

if __name__=='__main__':
    dh=np.array([-2,-1,0,1,1.2,1.4,1.8,2])
    loglog_v=dh2v_loglog(dh,160)
    quad_v=dh2v_quadratic(dh,160)
    # print(loglog_v)
    # print(quad_v)

    # print(v2dh_loglog(18,160))
    # print(dh2v_loglog(1.5,160))
    # print(v2dh_loglog(75,220))

    # print(v2dh_loglog(5,100))
    print(v2dh_loglog(6.6,200))
    print(v2dh_loglog(11.6,240))
    

    
    #print(v2dh_loglog(10,180))
    #print(v2dh_loglog(9.411764,160))
    # print(v2dh_loglog(6,100))
    # print(v2dh_loglog(10,100))
    # print(dh2v_loglog(5,100))
    # print(v2dh_quadratic(5,100))