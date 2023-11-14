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
            self.model_dir = properties.get("model_dir")
            self.gpu_id = properties.get("gpu_id")
            logger.info(f"System properties:{properties}")
            logger.info(f"Loading model artifacts from:{self.model_dir}. Directory contains{os.listdir(self.model_dir)}")
        except Exception as e:
            self.model_dir = './checkpoints/web_google-large-dino-20231011_T141010'
            # raise Exception(f"Failed Initialization")
        
            
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
    
    def preprocess(self, data, context):
        # Verify content type
        logger.info(f"Request data type: {type(data)}")
        if hasattr(context, 'request_content_type') and context.request_content_type != 'multipart/form-data':
            raise ValueError("Invalid content type, expected 'multipart/form-data'")
        # Extract and process the image under the key 'image'
        if data[0].get('image', None) is None:
            raise ValueError(
                "Missing image file under key 'image'")
        return [{
            'image': d.get("image"),
            'box_conf': float(
                d.get("box_conf", DEFAULT_CONF_THRES))
        } for d in data]
    
    def input_fn(self, data):
        """
        recieve list of data(json str). 
        Functions read image as array by decoding the byte string into utf-8 string , in which the data is decoded by base64 , read as 
        Input: 
            {'body': <base64 encoded string in byte/bytearray type>}
        
        """
        logger.info(f"Received request:{data}")
        inputs = []
        for n, d in enumerate(data):
            d['image'] = np.asarray(
                Image.open(io.BytesIO(d['image']))
            )
            inputs.append(d)
        return inputs

    def predict_fn(self, input_data, model=None):
        if not model: model = self.model
        if self.args is not None: conf=conf
        results = []
        with torch.no_grad():
            for d in input_data:
                r = model.predict(
                    d.get('image'), conf=d['box_conf'])
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
                r['labels'] = result.labels.tolist()
                r['probs']  = result.confidence.tolist()
            results.append(r)
        return results
        
    def handle(self, data, context):
        #   logger.info(f"Input data: \n{type(data),len(data)}\n, Context:\n{context.__dict__}")
        #   logger.info(f"Header Info:\n{context._request_processor[0]}")
      
      # Begin Handling 
        #   data = self.preprocess(data, context)
        x = self.input_fn(data)
        result = self.predict_fn(x)
        return self.output_fn(result)

_service = ModelHandler()

def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    return _service.handle(data, context)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start the API service.')
                        
    parser.add_argument('--loglevel', type=str, default='info',
                        choices=['debug', 'info', 'warning', 'error', 'critical'],
                        help='Logging level (default: info)')
    parser.add_argument('--image', type=str, default='../test-data/corn.jpg')
    parser.add_argument('--box_conf', type=str, default='../test-data/corn.jpg')
                        
    args = parser.parse_args()
    from flask import Flask, request, jsonify
    import requests
    app = Flask(__name__)

    @app.route('/predict', methods=['POST'])
    def post_example():
        # image = request.files.get('image')
        # box_conf = request.form.get('box_conf', type=float)
        resp = handle(request.data, request.method)
        # Just echoing the received values for demonstration purposes
        return jsonify(resp)
    
    # Testing the Flask app
    with app.test_client() as client:
        # Prepare the data for the POST request
        data = [{
            'image': (open(
                args.image, 'rb'), args.image),
            'box_conf': '0.5'
        }]

        # Send POST request
        response = client.post(
            '/predict', 
            content_type='multipart/form-data', 
            data=data)
        print(response.get_json())