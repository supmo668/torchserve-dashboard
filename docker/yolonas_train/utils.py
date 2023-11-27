import os, yaml
from os import path


def save_model_artifacts(model_artifacts_path, net):
    if path.exists(model_artifacts_path):
        model_file = open(model_artifacts_path + "model.dummy", "w")
        model_file.write("Dummy model.")
        model_file.close()


def print_files_in_path(path):
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            files.append(os.path.join(r, file))
    for f in files:
        print(f)

def get_data_conf(PATH_CONF):
    if hasattr(PATH_CONF, 'FETCH_DATA_ROBOFLOW') and FETCHPATH_CONF.FETCH_DATA_ROBOFLOW:
        import roboflow
        from roboflow import Roboflow
        roboflow.login()

        rf = Roboflow()
        rf.workspace().project_list
        project = rf.workspace().project(PATH_CONF.ROBOFLOW_PROJECT)
        dataset = project.version(1).download("yolov5")
        
        DATASET_PATH, CLASSES = dataset.location, sorted(project.classes.keys())
        EXPERIMENT_NAME = project.name.lower().replace(" ", "_")
    else:
        if PATH_CONF.DOWNLOAD_EACH_DATASET:
            downloadDirectoryFroms3(PATH_CONF.s3_bucket, PATH_CONF.s3_key)
            # CHECK dataset configuration
        assert PATH_CONF.DATASET_PATH.exists(), f"[ERROR] Dataset path at: {PATH_CONF.DATASET_PATH}\n do not exists. Either create or sync dataset to the location" 
        with open(PATH_CONF.DATASET_CONF_PATH, 'rb') as f: 
            DATA_CONF = yaml.safe_load(f)
        print(f"Data configuration\n:{DATA_CONF}")
        return DATA_CONF