import types, logging, time

import numpy as np
# from tensorflow.keras.applications import *

from ..utils.load_config import *

result_logger=logging.getLogger("vision_inference")
result_logger.setLevel(logging.DEBUG)

class classify_model:
    def __init__(self, 
        model_arch='mobilenetv3', input_shape=(224, 224, 3), weight:str='',
        label_cfg:dict={}, **kwargs
    ):
        """
        param
            preprocess (function): preprocess function
            class_indices (dict): numeric_label(int)[string_label(str)]
            conf_thres (int): confidence threshold , must be above or considered background
        """
        
        self.model = self.load_custom_tf_model(
            name=model_arch, n_classes=len(self.args['class_indices']), input_shape=input_shape,)
        kwargs['label_cfg'] = load_json(label_cfg)
        self.args = kwargs
        
    def load_custom_tf_model(
        self, name='mobilenetv3', n_classes:int = 3, input_shape=(224, 224, 3), 
        include_top = False):
        """Load TF models"""
        from tensorflow.keras.applications import MobileNetV3Large, MobileNetV2
        from tensorflow.keras import layers
        from tensorflow.keras.models import Model
        if name== 'mobilenetv3':
            model = MobileNetV3Large(
                input_shape=input_shape,
                include_top=include_top,
                weights='imagenet')
        elif name=='mobilenetv2':
            model = MobileNetV2(
                input_shape=input_shape,
                include_top=include_top,
                weights='imagenet' )
        else:
            raise Exception(f"model '{name}' not yet supported. Please add model to the model.py file")
        if not include_top: full_model = self.add_top_to_model(model, n_classes=n_classes)
        return full_model
        
    def add_top_to_model(self, model_bottom, n_classes=1):
        x = model_bottom.output
        x= layers.GlobalMaxPooling2D()(x)
        x = layers.Flatten()(x)
        x = layers.Dense(1024,activation='relu')(x) #dense layer 2
        x = layers.Dense(512,activation='relu')(x) #dense layer 3
        out = layers.Dense(n_classes, activation='softmax')(x) #final layer with softmax activation
        return Model(inputs=model_bottom.input, outputs=out)
    
    def infer(
        self, payload:dict, **kwargs) -> str:
        """
        param:
            payload (dict): dict containing image 'target'
        return
            string of item
        """
        # Preprocess image
        if not self.args['preprocess'] is None:
            img = self.args['preprocess'].get('function')(payload['target'].copy())   
        # predict
        p = self.model.predict(img[None, ...],verbose=0)
        if np.max(p) <= self.args['conf_thres']: return "bg"
        else: # map number back to class (e.g. 1: bean)
            result = {v:k for k, v in self.args['class_indices'].items()}[np.argmax(p)]
            return result if len(result)>0 else ['bg']
    @property
    def class_index(self):
        return self.args['class_indices']

def calculate_areas(boxes):
    arr = np.empty(0,)
    for xyxy in boxes:
        arr= np.append(arr, np.array(abs((xyxy[3]-xyxy[1])*(xyxy[2]-xyxy[0]))))
    return arr
  

class detect_model:
    def __init__(
        self, cfg_file:str='yolov8n.yaml', weights:str='yolov8n.pt', use_sagemaker:bool=False,
        dataset_cfg:str='grocer_eye.yaml', **kwargs):
        """
        parameters: 
             class_indices:dict={}, rank_criterion:str='box-cls', conf_thres: float=0.5, use_sagemaker=False, sagemaker_endpoint:str=''
        """
        if use_sagemaker:
            import boto3
            result_logger.info(
                'Using AWS sagemaker runtime for inference. Cancel in 5s to avoid additional costs.')
            self.model = boto3.Session().client('sagemaker-runtime')
            time.sleep(5)
        else:
            
            result_logger.info('Using local runtime for inference.')
            self.model = self.load_yolo(cfg_file=cfg_file, weights=weights)
        # load indices
        kwargs['class_indices'] = load_yaml_config_wobj(dataset_cfg)['names']
        self.args = kwargs
        
    def load_yolo(self, cfg_file:str, weights:str):
        from ultralytics import YOLO
        model = YOLO(weights)  # build from YAML and transfer weights
        return model
        
    def infer(self, payload:dict, top_k=None, aws_endpoint=False):
        """
        payload (dict): dict containing image 'target'
        top_k (int): return multiple 
        aws_endpoint (str): https://x.execute-api.us-east-2.amazonaws.com/main
        """
        if self.args.get('use_sagemaker'):
            response = self.model.invoke_endpoint(
                EndpointName= self.args.get('sagemaker_endpoint'),
                Body = payload['target'],
                ContentType='image/jpg'
            )
            result = response.get('Body').read().decode()
        else:
            output = self.model(payload['target'], stream=True, verbose=False)
            for r in output:    # return first result
                self.pred = r
                result = {
                    'box-coord': r.boxes.xyxy.cpu().numpy(),
                    'box-cls': r.boxes.cls.cpu().numpy(),
                    'box-area': calculate_areas(r.boxes.xyxy.cpu().numpy()),
                    'box-conf': r.boxes.conf.cpu(),
                    'masks': r.masks,
                    'probs': r.probs
                }
                continue
        # rank result (all fields) for selecting a scan
        rank = np.argsort(
            result[self.args.get('rank_criterion',self.args.get('box-conf'))])[::-1]
        if not top_k: top_k = min(1, len(r.boxes.cls))
        for k, v in result.items():
            # convert to list
            if not v is None and hasattr(v, '__len__'):
                result[k] = v[rank[:top_k].tolist()]
        # select by confidence
        top_items =[
            item for n, item in enumerate(result['box-cls']) if result['box-conf'][n] >= self.args['conf_thres']]
        if len(top_items) == 0: return ['bg']
        return [self.args['class_indices'][item_idx] for item_idx in top_items]
    
    @property
    def class_index(self):
        if hasattr(self, 'pred'):
            return self.pred.names
        else: 
            return {"error": "Uninitialized model inference."}
    
if __name__=="__main__":
    from vision.settings import vision_settings
    from vision.utils.load_config import *
    conf = load_yaml_config_wobj('config/default.yaml')
    settings=vision_settings(
        config_path=Path('config/default.yaml'), threaded=False
    ).settings
    detect_model