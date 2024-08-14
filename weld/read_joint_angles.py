import sys, glob, yaml
sys.path.append('../toolbox/')
sys.path.append('')
from lambda_calc import *
from multi_robot import *
from dx200_motion_program_exec_client import *
from WeldSend import *
from RobotRaconteur.Client import *
from weldRRSensor import *
from motoman_def import *
from flir_toolbox import *
import weld_dh2v
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.optimize import Bounds, minimize
from numpy.linalg import norm
import csv

robot=robot_obj('MA2010_A0',def_path='../config/MA2010_A0_robot_default_config.yml',tool_file_path='../config/torch.csv',\
	pulse2deg_file_path='../config/MA2010_A0_pulse2deg_real.csv',d=15)
robot2=robot_obj('MA1440_A0',def_path='../config/MA1440_A0_robot_default_config.yml',tool_file_path='../config/flir.csv',\
	pulse2deg_file_path='../config/MA1440_A0_pulse2deg_real.csv',base_transformation_file='../config/MA1440_pose.csv')
positioner=positioner_obj('D500B',def_path='../config/D500B_robot_extended_config.yml',tool_file_path='../config/positioner_tcp.csv',\
	pulse2deg_file_path='../config/D500B_pulse2deg_real.csv',base_transformation_file='../config/D500B_pose.csv')

H2010_1440=H_inv(robot2.base_H)
save_path = '../../recorded_data/test_heights/'

client=MotionProgramExecClient()
q = np.loadtxt(save_path+'q.csv',delimiter=',')
# q = np.zeros((4,8))
# for i in range(4):
#     input(f'jog to {i+1} and press enter')
#     q[i,:6] = client.getJointAnglesMH(robot.pulse2deg)
#     q[i,-2:] = client.getJointAnglesDB(positioner.pulse2deg)
# np.savetxt(save_path+'q.csv',q,delimiter=',')
print(np.rad2deg(q))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
height = []
for i in range(4):
    positioner_pose = positioner.fwd(q[i,-2:], world=True)
    robot1_pose = robot.fwd(q[i,:6],world=True)

    torch = positioner_pose.R.T@(robot1_pose.p-positioner_pose.p)

    ax.scatter(torch[0],torch[1],torch[2],'r')
    height.append(torch[2])
ax.set_aspect('equal')
plt.show()
print(np.average(height))
# average height is -13.33 mm