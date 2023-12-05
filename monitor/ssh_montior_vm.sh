#!/bin/bash
Help() {
  echo "Start up monitor locaclly via ssh tunnelling to torchserve dashbaord"
}
# Configuration
SSH_KEY_PATH=${1:"~/.ssh/gcp_ssh_key"}
GCP_INSTANCE_IP=${2:"35.245.161.246"}
SSH_USERNAME="mmym_ezout_gmail_com"


# Ports to tunnel (prometheus, grafana, torchserve-dashboard)
PORTS=("3030" "9090" "8105")
OpenSSHTunnel() {
  # Establish SSH tunnel for each port
  for port in "${PORTS[@]}"; do
      echo "Setting up tunnel for port $port"
      ssh -i "$SSH_KEY_PATH" -L "$port:localhost:$port" -N -f -l "$SSH_USERNAME" "$GCP_INSTANCE_IP"
  done

  echo "SSH tunnels established. Monitoring in progress..."
  
}
