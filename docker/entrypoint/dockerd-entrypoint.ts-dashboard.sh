#!/bin/bash
# modified torchserve-dashboard entrypointt
set -e
DASHBOARD_PORT=8105
echo "Mode used: $1"
if [[ "$1" = "serve" ]]; then
    shift 1
    echo "Serve Arguments: $@"
    torchserve-dashboard --server.port 8105 -- --config_path /home/model-server/config/config.ts-dashboard.properties --model_store /home/model-server/modelstore
elif [[ "$1" = "train" ]]; then
    shift 1
    echo "Train Arguments: $@"
    tensorboard --logdir=./checkpoint --port=6006 --bind_all &
    python yolonas_train/main.py "$@"
    # you may add: "-I datasets -C datasets_11202023" to the command line
else
    echo "This option is not currently supported. Arguments used: $@"
fi

# prevent docker exit
# tail -f /dev/null
