import RobotRaconteur as RR
RRN=RR.RobotRaconteurNode.s
import numpy as np
from flir_toolbox import *
import inspect, traceback, os, sys
sys.path.append("../toolbox/")
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
    def __init__(self, flir_service):

        self.flir_service = flir_service
        self.ir_image_consts = RRN.GetConstants('com.robotraconteur.image', self.flir_service)
        self.cam_pipe=self.flir_service.frame_stream.Connect(-1)
        self.cam_pipe.PacketRecievedEvent+=self.ir_cb
        try:
            self.flir_service.start_streaming()
        except:
            pass

        self.ir_process_struct=RRN.NewSturcutre("experimental.ir_process.ir_process_struct")
        self.flame_centroid_history = []
    
    def ir_cb(self,pipe_ep):
        while(pipe_ep.Available > 0):
            # read the image
            rr_img = pipe_ep.RecievePacket()
            if rr_image.image_invof.encoding == self.ir_image_consts["ImageEncoding"]["mono8"]:
                mat = rr_image.data.reshape([rr_img.image_info.height, rr_img.image_info.width], order='C')
            elif rr_image.image_invof.encoding == self.ir_image_consts["ImageEncoding"]["mono16"]:
                data_u16 = np.array(rr_img.data.view(np.uint16))
                mat = data_u16.reshape([rr_img.image_info.height, rr_img.image_info.width], order='C')
            ir_format = rr_img.image_info.extend["ir_format"].data

            if ir_format == "temperature_linear_10mK":
                display_mat = (mat*0.01) - 273.15
            elif ir_format == "temperature_linear_100mK":
                display_mat = (mat*0.1)-273.15
            else:
                display_mat = mat

            ir_image = np.rot90(display_mat, k=-1)
            centroid, bbox = flame_detection_aluminum()           
            if centroid is not None:
                center_x = centroid[0]
                center_y = centroid[1]+self.ir_pixel_window_size//2
                pixel_coord=(center_x, center_y)
                ## Not sure what this function does, need to track down
                flame_reading=get_pixel_value(ir_image, pixel_coord,self.ir_pixel_window_size)
                print(flame_reading, centroid)
                try:
                    self.ir_process_struct.flame_reading=int(flame_reading)
                    self.ir_process_struct.flame_position=centroid.astype(np.uint16)
                except:
                    traceback.print_exc()

if __name__ == '__main__':
    with RR.ServerNodeSetup("experimental.ir_process", 12182):
        flir_service=RRN.ConnectService('rr+tcp://localhost:60827/?service=camera')
        #Register the service type
        RRN.RegisterServiceType(ir_process)

        ir_process_obj=FLIR_RR_TRACKING(flir_service)

        #Regitser the service
        RRN.RegisterService("FLIR_RR_PROCESS", "experimental.ir_process.ir_process_obj", ir_process_obj)
        input("Press enter to quit")
