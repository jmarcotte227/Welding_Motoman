import RobotRaconteur as RR
RRN=RR.RobotRaconteurNode.s
import numpy as np
from flir_toolbox import *
from ultralytics import YOLO
import inspect, traceback, os

ir_process="""
service experimental.ir_process
struct ir_process_struct
	field uint16 flame_reading
	field uint16[] torch_bottom
	field uint16[] arc_centroid
end 
object ir_process_obj
	wire ir_process_struct ir_process_result [readonly]
end object
"""


class FLIR_RR_Process(object):
	def __init__(self,flir_service, torch_model, tip_wire_model):
		
		self.flir_service=flir_service
		self.ir_image_consts = RRN.GetConstants('com.robotraconteur.image', self.flir_service)
		# self.flir_service.setf_param("focus_pos", RR.VarValue(int(1900),"int32"))
		# self.flir_service.setf_param("object_distance", RR.VarValue(0.4,"double"))
		# self.flir_service.setf_param("reflected_temperature", RR.VarValue(291.15,"double"))
		# self.flir_service.setf_param("atmospheric_temperature", RR.VarValue(293.15,"double"))
		# self.flir_service.setf_param("relative_humidity", RR.VarValue(50,"double"))
		# self.flir_service.setf_param("ext_optics_temperature", RR.VarValue(293.15,"double"))
		# self.flir_service.setf_param("ext_optics_transmission", RR.VarValue(0.99,"double"))
		# self.flir_service.setf_param("current_case", RR.VarValue(2,"int32"))
		# self.flir_service.setf_param("ir_format", RR.VarValue("radiometric","string"))
		# self.flir_service.setf_param("object_emissivity", RR.VarValue(0.13,"double"))
		# self.flir_service.setf_param("scale_limit_low", RR.VarValue(293.15,"double"))
		# self.flir_service.setf_param("scale_limit_upper", RR.VarValue(5000,"double"))
		self.cam_pipe=self.flir_service.frame_stream.Connect(-1)
		#Set the callback for new pipe packets
		self.cam_pipe.PacketReceivedEvent+=self.ir_cb
		try:
			self.flir_service.start_streaming()
		except:
			pass
		
		#processing parameters
		self.ir_pixel_window_size=7
		self.torch_model = torch_model
		self.tip_wire_model = tip_wire_model
		self.ir_process_struct=RRN.NewStructure("experimental.ir_process.ir_process_struct")
		self.flame_centroid_history = []
		###bbox offset for pixel value from flame centroid
		self.vertical_offset=3
		self.horizontal_offset=0
		self.ir_pixel_window_size=7
	
			
	def ir_cb(self,pipe_ep):
		# Loop to get the newest frame
		while (pipe_ep.Available > 0):
			print("Received frame")
			# Receive the packet
			rr_img = pipe_ep.ReceivePacket()
			if rr_img.image_info.encoding == self.ir_image_consts["ImageEncoding"]["mono8"]:
				# Simple uint8 image
				mat = rr_img.data.reshape([rr_img.image_info.height, rr_img.image_info.width], order='C')
			elif rr_img.image_info.encoding == self.ir_image_consts["ImageEncoding"]["mono16"]:
				data_u16 = np.array(rr_img.data.view(np.uint16))
				mat = data_u16.reshape([rr_img.image_info.height, rr_img.image_info.width], order='C')

			ir_format = rr_img.image_info.extended["ir_format"].data

			if ir_format == "temperature_linear_10mK":
				display_mat = (mat * 0.01) - 273.15
			elif ir_format == "temperature_linear_100mK":
				display_mat = (mat * 0.1) - 273.15
			else:
				display_mat = mat

			ir_image = np.rot90(display_mat, k=-1)
			# centroid, bbox, torch_centroid, torch_bbox=weld_detection_aluminum(ir_image,self.yolo_model,percentage_threshold=0.8)
			centroid, bbox, torch_centroid, torch_bbox=weld_detection_steel(ir_image,self.torch_model,self.tip_wire_model)
			if centroid is not None:
				###weighted history filter
				if len(self.flame_centroid_history) > 30:
					self.flame_centroid_history.pop(0)
					# Calculate the weight for the previous history values
					previous_weight = 0.8 / len(self.flame_centroid_history)
					centroid = 0.2 * centroid + np.sum(np.array(self.flame_centroid_history) * previous_weight, axis=0)
					self.flame_centroid_history.append(centroid)

				#find average pixel value 
				pixel_coord = (int(centroid[0]) + self.horizontal_offset, int(centroid[1]) + self.vertical_offset)
				flame_reading=get_pixel_value(ir_image,pixel_coord,self.ir_pixel_window_size)

				print(flame_reading, torch_bbox, centroid)
				try:
					self.ir_process_struct.flame_reading=int(flame_reading)
					self.ir_process_struct.torch_bottom=np.array([torch_bbox[0]+torch_bbox[2]//2, torch_bbox[1]+torch_bbox[3]]).astype(np.uint16)
					self.ir_process_struct.arc_centroid=centroid.astype(np.uint16)
					self.ir_process_result.OutValue=self.ir_process_struct
				except:
					traceback.print_exc()

				


def main():
	#load model
	torch_model = YOLO(os.path.dirname(inspect.getfile(flir_toolbox))+"/torch.pt")
	tip_wire_model = YOLO(os.path.dirname(inspect.getfile(flir_toolbox))+"/tip_wire.pt")
	
	with RR.ServerNodeSetup("experimental.ir_process", 12182):
		flir_service=RRN.ConnectService('rr+tcp://localhost:60827/?service=camera')
		#Register the service type
		RRN.RegisterServiceType(ir_process)

		ir_process_obj=FLIR_RR_Process(flir_service, torch_model, tip_wire_model)
		
		#Register the service
		RRN.RegisterService("FLIR_RR_PROCESS","experimental.ir_process.ir_process_obj",ir_process_obj)
		input("Press enter to quit")
	
if __name__ == '__main__':
	main()
