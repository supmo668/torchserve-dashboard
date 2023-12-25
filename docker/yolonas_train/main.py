"""
Author: Matthew Mo
Organization: EzOut

This file assumes a standard YOLO dataset hosted on AWS S3 Bucket and the current host (
one that run the training) has the environment to be authorized to access the s3 bucket defined
in host credentials variables.
"""
from pathlib import Path
import os, yaml, json, datetime, shutil

import numpy as np
import supervision as sv, torch
from super_gradients.training import models, Trainer
from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_train, coco_detection_yolo_format_val
)
# from super_gradients.training.losses import PPYoloELoss
# from super_gradients.training.metrics import DetectionMetrics_050
# from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
import train_config

from visualize import plot_detections
from utils import get_data_conf

from log_config import LOGGING_CONFIG
import logging

logger = logging.getLogger(__name__)
logging.config.dictConfig(LOGGING_CONFIG)
logger.setLevel(logging.INFO)  # Set the logging level

# Meta Def
DEVICE = 'cuda' if torch.cuda.is_available() else "cpu"
HOME = Path.cwd()
datetime_str = datetime.datetime.now().strftime("%Y%m%d_T%H%M%S")
    # Local Attributes
DEFAULT_INPUT_HOME = Path(os.environ.get("DATASET_CHANNEL", './dataset'))
DEFAULT_CHECKPOINT_HOME = Path(os.environ.get('CHECKPOINT_HOME','./checkpoint')).absolute()
DEFAULT_SUBMIT_DIR = Path(os.environ.get('SUBMIT_DIRECTORY','/opt/ml/code'))

DEFAULT_MODEL_ARCH="yolo_nas_m"
INITIAL_LEARNING_RATE = 3e-5
# Configurations
class PATH_CONF:
    def __init__(self, args, INPUT_HOME=DEFAULT_INPUT_HOME, train_channel = 'train', **kwargs):
        """
        args: 
            {s3_bucket, s3_key,  s3_dataset_name}
            train_channel: sub_dir below 
        """
        self.__dict__.update(args.__dict__)
        
        self.DATASET_PATH = INPUT_HOME/train_channel
        self.DATASET_CONF_PATH = self.DATASET_PATH/'data.yaml'
        # DONWLOAD each subdataset (in addition to merging to one)
        self.DOWNLOAD_EACH_DATASET = False

class TRAIN_CONF:
    """
    Configuration of training along with validations built-in to the training loop
    """
    BATCH_SIZE:int = 8   # 
    NUM_WORKERS:int = 1 # Does not work > 2
    MAX_EPOCHS:int = 200
    def __init__(
        self, pretrained, model_arch, learning_rate=INITIAL_LEARNING_RATE, 
        checkpoint=None, job_name='ezout-vision-train',
        **kwargs
        ):
        self.__dict__.update(args.__dict__) 
        model = models.get(
            self.model_arch, 
            pretrained_weights= "coco" if pretrained else False
        ).to(DEVICE)
        self.CHECKPOINT_DIR = DEFAULT_CHECKPOINT_HOME
        self.LR = learning_rate
        # Load from checkpoint (path to .pth )
        self.FROM_CHECKPOINT = checkpoint
        if not checkpoint:
            self.EXPERIMENT_NAME = f"{job_name}T{datetime_str}"
        else:
            # use the name of directory to the path of checkpoint provided
            self.EXPERIMENT_NAME = Path(checkpoint).parent.name  
        self.CHECKPOINT_OUTPUT_DIR = self.CHECKPOINT_DIR/self.EXPERIMENT_NAME
        self.CHECKPOINT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Checkpoint output directory: {self.CHECKPOINT_OUTPUT_DIR}")
        # Testing
        self.CONFIDENCE_THRESHOLD = 0.3
        
def createDataset(PATH_CONF, TRAIN_CONF):
    ####### Datasets
    data_conf = get_data_conf(PATH_CONF)
    logger.info(f"Creating dataset from: {PATH_CONF.DATASET_PATH}")
    datasets = dict()
    for tt in [sp for sp in ['train', 'test', 'valid'] if data_conf.get(sp)]:
        # Check for existence of split
        logger.info(f"Loading splits: {tt}.")
        logger.info(
            f"image:{PATH_CONF.DATASET_PATH/Path(data_conf[tt])}\n" \
            f"labels:{(PATH_CONF.DATASET_PATH/Path(data_conf[tt]))/'labels'}\n" \
            f"class:{data_conf.get('names')}"
        )
        if data_conf.get(tt):
            if tt=='train': create_dataset_func = coco_detection_yolo_format_train
            else: create_dataset_func = coco_detection_yolo_format_val   
            datasets[tt] = create_dataset_func(
                dataset_params={
                    'data_dir': PATH_CONF.DATASET_PATH,
                    'images_dir': (PATH_CONF.DATASET_PATH/Path(data_conf[tt])).resolve(),
                    'labels_dir': (PATH_CONF.DATASET_PATH/Path(data_conf[tt])).resolve().parent/'labels',
                    'classes': data_conf.get('names')
                },
                dataloader_params={
                    'batch_size': TRAIN_CONF.BATCH_SIZE,
                    'num_workers': TRAIN_CONF.NUM_WORKERS
                }
            )
    logger.info(f"found {list(datasets.keys())}: dataset")
    return datasets

def train(PATH_CONF, TRAIN_CONF):
    '''Initialize trainer'''
    data_conf = get_data_conf(PATH_CONF)
    # COPY data yaml file into checkpoint directory for packaging purpose
    if (TRAIN_CONF.CHECKPOINT_DIR/PATH_CONF.DATASET_CONF_PATH.name).exists():
        logger.info(f"[Warning]{PATH_CONF.DATASET_CONF_PATH.name} already exists in \
            {TRAIN_CONF.CHECKPOINT_DIR}. The previous data YAML will be overwritten.")
    shutil.copy(
        PATH_CONF.DATASET_CONF_PATH, TRAIN_CONF.CHECKPOINT_OUTPUT_DIR)
    trainer = Trainer(
        experiment_name=TRAIN_CONF.EXPERIMENT_NAME, 
        ckpt_root_dir=TRAIN_CONF.CHECKPOINT_DIR
    )
    datasets = createDataset(PATH_CONF, TRAIN_CONF)
    ######## Model
    if TRAIN_CONF.FROM_CHECKPOINT:
        logger.info(f"Training from checkpoint: {TRAIN_CONF.FROM_CHECKPOINT}")
        model = models.get(
            TRAIN_CONF.model_arch,
            num_classes=len(data_conf.get('names')),
            checkpoint_path=str(TRAIN_CONF.FROM_CHECKPOINT)
        ).to(DEVICE)
    else:
        logger.info(f"Training from pre-trained: {TRAIN_CONF.model_arch}")
        model = models.get(
            TRAIN_CONF.model_arch, 
            num_classes=len(data_conf.get('names')), 
            pretrained_weights="coco"
        ).to(DEVICE)

    # TRAINING params
    trainer.train(
        model=model, 
        training_params=train_config.get_training_parameters(
            TRAIN_CONF, len(data_conf.get('names'))
        ), 
        train_loader=datasets['train'], 
        valid_loader=datasets['valid']
    )
    return trainer
    
def evaluate(
    trainer, datasets, path_conf, train_conf, best_model="ckpt_best.pth"):
    data_conf = get_data_conf(path_conf)
    #####  Evaluate
    if not hasattr(path_conf, 'FROM_CHECKPOINT'):
        path_conf.FROM_CHECKPOINT = train_conf.CHECKPOINT_OUTPUT_DIR/best_model
        assert path_conf.FROM_CHECKPOINT.exists(), f"Checkpoint path: {path_conf.FROM_CHECKPOINT} don't exists."
    else:
        assert path_conf.FROM_CHECKPOINT.exists(), f"Using checkpoint {path_conf.FROM_CHECKPOINT} but not exists"
        logger.info(f"Evaluating from checkpoint {path_conf.FROM_CHECKPOINT}")
    best_model = models.get(
        train_conf.model_arch,
        num_classes=len(data_conf.get('names')),
        checkpoint_path=str(path_conf.FROM_CHECKPOINT)).to(DEVICE)

    if not (data_conf.get('test') and data_conf.get('valid')):
        logger.info(f"Both test & valid test set is not available")
        return 
    else:
        if data_conf.get('test'): 
            test_set = datasets['test']
        elif data_conf.get('valid'): 
            test_set = datasets['valid']
    
    from onemetric.cv.object_detection import ConfusionMatrix
    evaluation = trainer.test(
        model=best_model,
        test_loader=test_set,
        test_metrics_list=train_config.DetectionMetrics_050(
            score_thres=0.1, 
            top_k_predictions=300, 
            num_cls=len(data_conf.get('names')),
            normalize_targets=True, 
            post_prediction_callback=train_config.PPYoloEPostPredictionCallback(
                score_threshold=0.01, 
                nms_top_k=1000, 
                max_predictions=300,
                nms_threshold=0.7
            )
        )
    )
    evaluation = {
        k:float(v) for k,v in evaluation.items()}
    with open(
        train_conf.CHECKPOINT_OUTPUT_DIR/'model_evaluation_metric.json', 'w') as fp:
        json.dump(evaluation, fp)
    ds_test = sv.DetectionDataset.from_yolo(
        images_directory_path=(path_conf.DATASET_CONF_PATH/Path(data_conf.get(test_set))).resolve(),
        annotations_directory_path=(path_conf.DATASET_CONF_PATH/Path(data_conf.get(test_set))).resolve().parent/'labels',
        data_yaml_path=path_conf.DATASET_CONF_PATH,
        force_masks=False
    )
    logger.info(f"Dataset size:\n{len(test_set)}")
    ### Create predictions 
    predictions = {}
    for image_name, image in test_set.ds_test.items():
        result = list(best_model.predict(image, conf=train_conf.CONFIDENCE_THRESHOLD))[0]
        detections = sv.Detections(
            xyxy=result.prediction.bboxes_xyxy,
            confidence=result.prediction.confidence,
            class_id=result.prediction.labels.astype(int)
        )
        predictions[image_name] = detections
    logger.info(f"Predictions:\n{predictions}")
    keys = list(test_set.images.keys())
    annotation_batches, prediction_batches = [], []
    ### Sort prediction by keys
    for key in keys:
        annotation=test_set.annotations[key]
        annotation_batch = np.column_stack((
            annotation.xyxy, 
            annotation.class_id
        )) 
        annotation_batches.append(annotation_batch)
        #
        prediction=predictions[key]
        prediction_batch = np.column_stack((
            prediction.xyxy, 
            prediction.class_id,
            prediction.confidence
        ))
        prediction_batches.append(prediction_batch)
        
    confusion_matrix = ConfusionMatrix.from_detections(
        true_batches=annotation_batches, 
        detection_batches=prediction_batches,
        num_classes=len(test_set.classes),
        conf_threshold=train_conf.CONFIDENCE_THRESHOLD
    )
    confusion_matrix.plot(
        train_conf.CHECKPOINT_OUTPUT_DIR/"confusion_matrix.png",
        class_names=test_set.classes
    )
    logger.info(f"Evaluations saved to checkpoint directory:\n{train_conf.CHECKPOINT_OUTPUT_DIR}")
        
def main(args):
    path_conf = PATH_CONF(
        args, args.input, args.train_channel) 
    train_conf = TRAIN_CONF(
        pretrained=args.pretrained, model_arch=args.model_arch, learning_rate=args.lr,
        checkpoint=args.chkpt, job_name=args.train_channel)   #
    datasets = createDataset(path_conf, train_conf)
    if not args.val_only:
        trainer = train(path_conf, train_conf)
    else:
        logger.info("Running validations only")
        trainer = Trainer(
            experiment_name=train_conf.EXPERIMENT_NAME, 
            ckpt_root_dir=train_conf.CHECKPOINT_DIR
        )
    evaluate(trainer, datasets, path_conf, train_conf)
    
if "__main__"==__name__:
    """
    CMD Entrypoint for running training
    arguments:
        input           : outer path to dataset directories (with multiple dataset)
        train_channel   : dataset name 
    Example:
        python main-local.py -I ../datasets -C my_dataset
    """
    import argparse 
    parser = argparse.ArgumentParser()
    # reads input channels training and testing from the environment variables
    parser.add_argument(
        "-I", "--input", type=Path, default=DEFAULT_INPUT_HOME)
    parser.add_argument(
        "-C", "--train_channel", type=str, default="train")
    parser.add_argument(
        "--model_arch", type=str, default=DEFAULT_MODEL_ARCH)
    parser.add_argument(
        "--pretrained", type=bool, default=True)
    parser.add_argument(
        "--val_only", type=bool, default=False)
    parser.add_argument(
        "--chkpt", type=Path, default=None)
    parser.add_argument(
        "-LR", "--lr", type=float, help="Initial learning rate", 
        default=INITIAL_LEARNING_RATE)
    args = parser.parse_args()
    main(args)