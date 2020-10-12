import os

import cv2
import cv2 as cv
import json
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.modeling import build_model
import torch
import numpy as np
from PIL import Image

from gradient_utils.metrics import MetricsLogger

try:
	logger = MetricsLogger()
	logger.add_counter("inference_count")
except:
	logger = None


class Detector:

	def __init__(self):

		# set model and test set
		self.model = 'mask_rcnn_R_50_FPN_3x.yaml'

		# obtain detectron2's default config
		self.cfg = get_cfg() 

		# load values from a file
		# self.cfg.merge_from_file("test.yaml")
		self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/"+self.model)) 

		# Additional Info when using cuda
		if torch.cuda.is_available():
			self.cfg.MODEL.DEVICE = "cuda"
			print(torch.cuda.get_device_name(0), flush=True)
			print('Memory Usage:', flush=True)
			print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB', flush=True)
			print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB', flush=True)
		else:
			# set device to cpu
			self.cfg.MODEL.DEVICE = "cpu"

		# get Model
		model_path = os.path.abspath('/models/model/detectron/model_final.pth')
		if os.path.isfile(model_path):
			print('Using Trained Model {}'.format(model_path), flush=True)
		else:
			print('No Model Found at {}'.format(model_path), flush=True)
			model_path = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/"+self.model)
		self.cfg.MODEL.WEIGHTS = model_path
		# set the testing threshold for this model
		self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  


	# build model and convert for inference
	def convert_model_for_inference(self):

		# build model
		model = build_model(self.cfg)

		# save as checkpoint
		torch.save(model.state_dict(), 'checkpoint.pth')

		# return path to inference model
		return 'checkpoint.pth'

	# detectron model
	def inference(self, file):

		predictor = DefaultPredictor(self.cfg)
		im = cv.imread(file)
		rgb_image = im[:, :, ::-1]
		outputs = predictor(rgb_image)

		# with open(self.curr_dir+'/data.txt', 'w') as fp:
		# 	json.dump(outputs['instances'], fp)
		# 	# json.dump(cfg.dump(), fp)

		# get metadata
		metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
		# visualise
		v = Visualizer(rgb_image[:, :, ::-1], metadata=metadata, scale=1.2)
		v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

		# get image 
		img = Image.fromarray(np.uint8(v.get_image()[:, :, ::-1]))

		if logger:
			# Push Metrics
			logger["inference_count"].inc()
			logger.push_metrics()
			print('Logged Inference Count', flush=True)

		return img



