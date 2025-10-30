import numpy as np
import sys, traceback, time, copy, glob
from general_robotics_toolbox import *
import matplotlib.pyplot as plt



def main():
    dataset='wall/'
    sliced_alg='1_5mm_slice/'
    data_dir='../data/'+dataset+sliced_alg
    num_layers=29
    num_baselayers=0
    num_sections = 1
    curve_sliced_js=[]
    positioner_js=[]
    rob_js=[]
    rob_2_js=[]
    for i in range(num_layers):
        # num_sections=len(glob.glob(data_dir+'curve_sliced_js/D500B_js'+str(i)+'_*.csv'))
        for x in range(num_sections):
            if i % 2 == 0:
                # curve_sliced_js.append(np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(i)+'_'+str(x)+'.csv',delimiter=','))
                positioner_js.append(np.loadtxt(
                    data_dir+'curve_sliced_js/D500B_js'+str(i)+'_'+str(x)+'.csv',
                    delimiter=','
                ))
                rob_js.append(np.loadtxt(
                    data_dir+'curve_sliced_js/MA2010_js'+str(i)+'_'+str(x)+'.csv',
                    delimiter=','
                ))
                rob_2_js.append(np.loadtxt(
                    data_dir+'curve_sliced_js/MA1440_js'+str(i)+'_'+str(x)+'.csv',
                    delimiter=','
                ))
            
            else:
                # curve_sliced_js.append(np.flip(np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(i)+'_'+str(x)+'.csv',delimiter=','),axis=0))
                positioner_js.append(np.flip(np.loadtxt(
                    data_dir+'curve_sliced_js/D500B_js'+str(i)+'_'+str(x)+'.csv',
                    delimiter=','
                ),axis=0))
                rob_js.append(np.flip(np.loadtxt(
                    data_dir+'curve_sliced_js/MA2010_js'+str(i)+'_'+str(x)+'.csv',
                    delimiter=','
                ),axis=0))
                rob_2_js.append(np.flip(np.loadtxt(
                    data_dir+'curve_sliced_js/MA1440_js'+str(i)+'_'+str(x)+'.csv',
                    delimiter=','
                ),axis=0))

    # curve_sliced_js=np.concatenate( curve_sliced_js, axis=0)
    positioner_js=np.concatenate( positioner_js, axis=0 )
    rob_js=np.concatenate(rob_js, axis=0 )
    rob_2_js=np.concatenate(rob_2_js, axis=0 )
    print(rob_js.shape)

    plt.plot(positioner_js,label=('q1','q2'))
    plt.legend()
    plt.title(' '+str(num_layers)+' slices')
    plt.show()
    plt.plot(rob_js,label=('q1','q2','q3','q4', 'q5','q6'))
    plt.legend()
    plt.title(' '+str(num_layers)+' slices')
    plt.show()
    plt.plot(rob_2_js,label=('q1','q2','q3','q4', 'q5','q6'))
    plt.legend()
    plt.title(' '+str(num_layers)+' slices')
    plt.show()



if __name__ == '__main__':
    main()
