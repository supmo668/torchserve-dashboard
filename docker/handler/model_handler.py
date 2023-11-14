"""
ModelHandler defines an example model handler for load and inference requests for MXNet CPU models
"""
import logging, time
from pathlib import Path

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    filename=Path.home()/'app.log')
print(f"Logger is logged to:{Path.home()/'app.log'}")

import os, yaml, io, base64, json, logging, argparse
from PIL import Image

import numpy as np
import torch
from super_gradients.training import models

# Initialize logging
logger = logging.getLogger(__name__)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_MODEL_DIR = "/home/model-server/modelstore"
DEFAULT_MODEL_ARCH = "YOLO_NAS_M"
DEFAULT_CONF_THRES = 0.3

# Required file:
    # model_dir/'data.yaml'
    # model_dir/'ckpt_best.pth
    
class ModelHandler(object):
    """Handler service that is executed by the model server.
    Determines specific default inference handlers to use based on model being used.
    This class extends ``DefaultHandlerService``, which define the following:
        - The ``handle`` method is invoked for all incoming inference requests to the model server.
        - The ``initialize`` method is invoked at model server start up.
    Based on: https://github.com/awslabs/mxnet-model-server/blob/master/docs/custom_service.md
    """
    def __init__(self, args=None, **kwargs):
        
        """
        SageMaker Model handler for YOLO vision inference.
        args (params):
            model_arch
            checkpoint_dir
            model_name
        """
        self.initialized = False
        self.args = args
        self.kwargs = kwargs

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        self.initialized = True
        try:
            self.context = context
            properties = context.system_properties
            # Contains the url parameter passed to the load request
            self.model_dir = properties.get(
                "model_dir")
            self.gpu_id = properties.get("gpu_id")
            logger.info(f"System properties:{properties}")
            logger.info(f"Loading model artifacts from:{self.model_dir}. Directory contains{os.listdir(self.model_dir)}")
        except Exception as e:
            self.model_dir = self.kwargs['model_dir']
            logger.debug(f"Failed Initialization")
        
        self.model = self.model_fn(
            self.model_dir, 'ckpt_best.pth')
    
    def model_fn(self, model_dir, pth_name):
        """
        fetch and load best model for deployment
        """
        assert (Path(model_dir)/'data.yaml').exists(), "Please provide data.yaml to facilitate inference"
        with open(Path(model_dir)/'data.yaml') as f:
            data_conf = yaml.safe_load(f)
        logger.info(f"Loading model from {model_dir}")
        model = models.get(
            os.environ.get('model_arch', DEFAULT_MODEL_ARCH),
            num_classes=len(data_conf.get('names')),
            checkpoint_path=str(Path(model_dir)/pth_name)
        )
        model.to(device)
        return model
    
    def input_fn(self, data):
        """
        recieve list of data(json str). 
        Functions read image as array by decoding the byte string into utf-8 string , in which the data is decoded by base64 , read as 
        Input: 
            {'body': <base64 encoded string in byte/bytearray type>}
        
        """
        logger.info(f"Received request data.")
        inputs = []
        payload_dtypes = {k: type(v) for k,v in data[0].items()}
        logger.info(f"Payload data types: {payload_dtypes}")
        try:
            for n, d in enumerate(data):
                d['image'] = np.asarray(
                    Image.open(io.BytesIO(d['image']))
                )
                inputs.append(d)
        except Exception as e:
            return f"[Error]:{e}"
        return inputs

    def predict_fn(self, input_data, model=None):
        logger.info(f"Received prediction input data")
        if not model: model = self.model
        if self.args is not None: conf=conf
        results = []
        with torch.no_grad():
            for d in input_data:
                r = model.predict(
                    d.get('image'), conf=float(d['box_conf']))
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
        
    def handle(self, data, context):
        x = self.input_fn(data)
        result = self.predict_fn(x)
        return self.output_fn(result)

_service = ModelHandler()

def handle(data, context, **kwargs):
    if not _service.initialized:
        logger.info(f"Initializing service.")
        _service.initialize(context)

    if data is None:
        return None
    logger.info(f"Handle data.")
    return _service.handle(data, context)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start the API service.')
                        
    parser.add_argument(
        '--loglevel', type=str, default='info',
        choices=['debug', 'info', 'warning', 'error', 'critical'],
        help='Logging level (default: info)')
    parser.add_argument(
        '--model_dir', type=str, default="../modelstore/web_google-large-dino-20231011_T141010/")
    parser.add_argument(
        '--image', type=str, default='../test-data/corn.jpg')
    args = parser.parse_args()
    _service = ModelHandler(model_dir=args.model_dir)
    with open(args.image, 'rb') as f:
        img_byte = f.read()
    data = [{
        'image': img_byte,
        'box_conf': '0.5'
    }]
    result = handle(data=data, context=None)
    print(f"output:\n{result}")
