#!/bin/bash
Help() {
  echo "export model from .pth to .mar in the same directory."
  echo "Arg 1    : Name of directory of the location of model in checkpoint directory "
  echo "[ELEVATED permission]: sudo -u mmym_ezout_gmail_com env PATH=$PATH bash torch-archiv-export.sh  [args]"
  
}
# HARD-CODED
ckpt_file="ckpt_best.pth"
default_handler="$PWD/handler/model_handler:handle"
EXPORT_PATH="modelstore"
# soft-coded
MODEL=${1:-web_google-large-dino-20231011_T141010}
MODEL_CKPTS=${2:-"./checkpoint"}
HANDLER=${3:-$default_handler}
VERSION=${4:-1.0.0}
# script
MODEL_PATH=$MODEL_CKPTS/$MODEL/$ckpt_file
mkdir -p $EXPORT_PATH
chmod 600 $EXPORT_PATH
echo "Creating model archiv from:..."
echo "    $MODEL_CKPTS"
echo "    $HANDLER"
echo "    $(dirname "$MODEL_PATH")/data.yaml"
echo "exporting to: $export_path"
torch-model-archiver -f \
  --model-name=$MODEL \
  --version $VERSION \
  --serialized-file="$MODEL_PATH" \
  --handler="$HANDLER" \
  --extra-files "$(dirname "$MODEL_PATH")/data.yaml" \
  --export-path=$EXPORT_PATH


# torch-model-archiver --model-name web_google-large-dino-20231011_T141010 \
# --version 1.0  --serialized-file checkpoints/web_google-large-dino-20231011_T141010/ckpt_best.pth --extra-files checkpoints/web_google-large-dino-20231011_T141010/data.yaml  --handler mms_ezout/model_handler.py --export-path=modelstore
  
#  `$(basename $(dirname "$MODEL_PATH"))`