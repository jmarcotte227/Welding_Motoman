import RobotRaconteur as RR
RRN=RR.RobotRaconteurNode.s
import numpy as np
from flir_toolbox import *
import inspect, traceback, os

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

