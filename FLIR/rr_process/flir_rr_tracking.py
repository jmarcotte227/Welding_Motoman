import RobotRaconteur as RR
RRN=RR.RobotRaconteurNode.s
import numpy as np
from flir_toolbox import *
from motoman_def import robot_obj, positioner_obj
import inspect, traceback, os, sys
import matplotlib.pyplot as plt
sys.path.append("../../toolbox/")
sys.path.append("")
from angled_layers import flame_detection_aluminum

ir_process="""
service experimental.ir_process
struct ir_process_struct
    field uint16 pixel_reading
    field uint16[] flame_position
end
object ir_process_obj
    wire ir_process_struct ir_process_result [readonly]
end object
"""

class FLIR_RR_TRACKING(object):
    def __init__(self, flir_service, robot_service):

        self.flir_service = flir_service
        self.robot_service = robot_service
        self.ir_image_consts = RRN.GetConstants('com.robotraconteur.image', self.flir_service)
        self.cam_pipe=self.flir_service.frame_stream.Connect(-1)
        self.cam_pipe.PacketReceivedEvent+=self.ir_cb
        try:
            self.flir_service.start_streaming()
        except:
            print("unable to start streaming")

        self.ir_process_struct=RRN.NewStructure("experimental.ir_process.ir_process_struct")
        self.flame_centroid_history = []

        ######## ROBOTS ########
        # Define Kinematics
        CONFIG_DIR = '../../config/'
        self.robot=robot_obj(
            'MA2010_A0',
            def_path=CONFIG_DIR+'MA2010_A0_robot_default_config.yml',
            tool_file_path=CONFIG_DIR+'torch.csv',
            pulse2deg_file_path=CONFIG_DIR+'MA2010_A0_pulse2deg_real.csv',
            d=15
        )
        self.robot2=robot_obj(
            'MA1440_A0',
            def_path=CONFIG_DIR+'MA1440_A0_robot_default_config.yml',
            tool_file_path=CONFIG_DIR+'flir.csv',
            pulse2deg_file_path=CONFIG_DIR+'MA1440_A0_pulse2deg_real.csv',
            base_transformation_file=CONFIG_DIR+'MA1440_pose.csv'
        )
        self.positioner=positioner_obj(
            'D500B',
            def_path=CONFIG_DIR+'D500B_robot_extended_config.yml',
            tool_file_path=CONFIG_DIR+'positioner_tcp.csv',
            pulse2deg_file_path=CONFIG_DIR+'D500B_pulse2deg_real.csv',
            base_transformation_file=CONFIG_DIR+'D500B_pose.csv'
        )
    # initialize display
        self.fig, self.ax = plt.subplots()
        self.ax.imshow(np.zeros((480,640)))
        plt.show()


    def ir_cb(self,pipe_ep):
        '''
        Processes the ir frames as they come in and maps them to joint angles
        '''
        while pipe_ep.Available > 0:
            # read the joint angles first
            q_cur=self.robot_service.robot_state.PeekInValue()[0].joint_position
            # print("q_cur: ", q_cur)
            # TODO: Check and see if this is actually quick enough to be accurate

            # read the image
            rr_img = pipe_ep.ReceivePacket()

            if rr_img.image_info.encoding == self.ir_image_consts["ImageEncoding"]["mono8"]:
                mat = rr_img.data.reshape(
                    [rr_img.image_info.height, rr_img.image_info.width],
                    order='C'
                )
            elif rr_img.image_info.encoding == self.ir_image_consts["ImageEncoding"]["mono16"]:
                data_u16 = np.array(rr_img.data.view(np.uint16))
                mat = data_u16.reshape(
                    [rr_img.image_info.height, rr_img.image_info.width],
                    order='C'
                )
            ir_format = rr_img.image_info.extended["ir_format"].data

            if ir_format == "temperature_linear_10mK":
                display_mat = (mat*0.01) - 273.15
            elif ir_format == "temperature_linear_100mK":
                display_mat = (mat*0.1)-273.15
            else:
                display_mat = mat

            ir_image = np.rot90(display_mat, k=-1)
            self.ax.imshow(ir_image)
            self.fig.canvas.draw()
            centroid, _ = flame_detection_aluminum(ir_image)
            print("centroid: ", centroid)
            if centroid is not None:
                center_x = centroid[0]
                center_y = centroid[1]
                pixel_coord=(center_x, center_y)
                ## Not sure what this function does, need to track down
                flame_reading=get_pixel_value(ir_image, pixel_coord,self.ir_pixel_window_size)
                print(flame_reading, centroid)
                try:
                    self.ir_process_struct.flame_reading=int(flame_reading)
                    self.ir_process_struct.flame_position=centroid.astype(np.uint16)
                    self.ir_process_result.OutValue=self.ir_process_struct
                except:
                    traceback.print_exc()

if __name__ == '__main__':
    with RR.ServerNodeSetup("experimental.ir_process", 12182):
        flir_service=RRN.ConnectService('rr+tcp://localhost:60827/?service=camera')
        robot_service = RRN.ConnectService('rr+tcp://localhost:59945?service=robot')
        #Register the service type
        RRN.RegisterServiceType(ir_process)

        ir_process_obj=FLIR_RR_TRACKING(flir_service, robot_service)

        #Regitser the service
        RRN.RegisterService("FLIR_RR_PROCESS", "experimental.ir_process.ir_process_obj", ir_process_obj)
        input("Press enter to quit")
