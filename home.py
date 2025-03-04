import sys
sys.path.append('toolbox/')
from motoman_def import *
from WeldSend import *
from dx200_motion_program_exec_client import *

config_dir='config/'
robot_weld=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',d=15,tool_file_path=config_dir+'torch.csv',\
	pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv')
robot_scan=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'mti.csv',\
	base_transformation_file=config_dir+'MA1440_pose.csv',pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv')
positioner=positioner_obj('D500B',def_path=config_dir+'D500B_robot_default_config.yml',tool_file_path=config_dir+'positioner_tcp.csv',\
    base_transformation_file=config_dir+'D500B_pose.csv',pulse2deg_file_path=config_dir+'D500B_pulse2deg_real.csv')


q1=np.array([0,0,0,0,0,0])
q2=np.array([90,0,0,0,0,0])
# q2=np.array([0,37,-9,0,0,0])
# q2_1=np.array([0,36,-8,0,0,0])
q3=[-15,0]

robot_client=MotionProgramExecClient()
ws=WeldSend(robot_client)
ws.jog_single(positioner,np.radians([-15,0]),v=100)
ws.jog_single(robot_weld,np.radians(q1),v=5)
ws.jog_single(robot_scan,np.radians(q2),v=5)
