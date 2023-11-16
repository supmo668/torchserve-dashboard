#!/bin/bash
# modified torchserve-dashboard entrypointt
set -e
DASHBOARD_PORT=8105
if [[ "$1" = "serve" ]]; then
    shift 1
    torchserve-dashboard --server.port 8105 -- --config_path /home/model-server/config/config.ts-dashboard.properties --model_store /home/model-server/model-store
else
    eval "$@"
fi

# prevent docker exit
tail -f /dev/null
