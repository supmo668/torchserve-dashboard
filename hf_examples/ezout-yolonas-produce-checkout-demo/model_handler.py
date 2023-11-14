from typing import Dict, List, Any

import os, yaml, io, base64, json, logging, argparse
from PIL import Image

from pathlib import Path
import numpy as np
import torch, torchaudio
from super_gradients.training import models

import logging
import logging.config

# Define logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'level': 'DEBUG',
            'stream': 'ext://sys.stdout',
        },
        'file': {
            'class': 'logging.FileHandler',
            'formatter': 'standard',
            'level': 'INFO',
            'filename': 'api_server.log',
        },
    },
    'loggers': {
        '': {  # root logger
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': True,
        },
        'my_api': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False,
        },
    }
}

# Apply the logging configuration
logging.config.dictConfig(LOGGING_CONFIG)

# Usage example
logger = logging.getLogger('my_api')
logger.info("API server started")

DEFAULT_MODEL_ARCH = "YOLO_NAS_M"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
class yolonas_predictor:
  def __init__(self, modelstore="modelstore"):
    self.model = self.model_fn(
      modelstore, 'ckpt_best.pth')

  
  def model_fn(self, model_dir, name):
    """fetch a load model"""
    assert (Path(model_dir)/'data.yaml').exists(), "Please provide data.yaml to facilitate inference"
    with open(Path(model_dir)/'data.yaml') as f:
      data_conf = yaml.safe_load(f)
    logger.info(f"Loading model from {model_dir}")
    model = models.get(
      os.environ.get('model_arch', DEFAULT_MODEL_ARCH),
      num_classes=len(data_conf.get('names')),
      checkpoint_path=str(Path(model_dir)/name)
    )
    model.to(DEVICE)
    return model

  @torch.no_grad()
  def predict_fn(self, input_data, model=None):
    logger.info(f"Received prediction input data")
    if not model: model = self.model
    if self.args is not None: conf=conf
    results = []
    with torch.no_grad():
      for d in input_data:
        r = model.predict(
          d.get('image'), conf=float(d['box_conf'])
          )
        results.append(r)
    return results
    
  def output_fn(
    self, prediction_output, content_type=None):
        # probs_sort = np.argsort(probs)
        logger.info(f"Predicting results.")
        results = []
        for out in prediction_output:
            r = {}
            for p in out:
                result = p.prediction
                r['boxes'] = result.bboxes_xyxy.tolist()
                r['labels'] = result.labels.astype('int').tolist()
                r['probs']  = result.confidence.tolist()
            results.append(r)
        return results
      
class EndpointHandler:
	def __init__(self, modelstore):
		# load model and processor from path
		self.predictor = yolonas_predictor(Path(modelstore))

	def __call__(self, data: Dict[str, Any]) -> Dict[str, str]:
		"""
		Args:
				data (:dict:):
						The payload with the text prompt and generation parameters.
		"""
		# prompt_duration = 2
		# process input
		inputs = data.pop('inputs')
		img = inputs.pop("image", inputs)
		box_conf = inputs.pop("box_conf", inputs)
		print(type(img))
		payload = np.asarray(Image.open(io.BytesIO(img)))

		output = self.predictor.predict_fn(
    	[{'image':img,'box_conf': box_conf}])
		
		# postprocess the prediction
		prediction = self.predictor.output_fn(output)

		return [{"inference": prediction}]
