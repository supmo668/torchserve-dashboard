Help() {
  echo "Add monitoring firewall rules to allow ssh monitoring of the VM in GCP"
}

INSTANCE_NAME=${1:-"dl-instance"}
PROJECT_NAME=${2:-"ezout-vision-402700"}
AddFirewallRule() {
  gcloud compute instances add-tags $INSTANCE_NAME --tags=torchserve --zone=us-east4-c  --project $PROJECT_NAME
  gcloud compute firewall-rules create allow-ssh-custom-ports \
    --direction=INGRESS \
    --priority=1000 \
    --network=default \
    --action=ALLOW \
    --rules=tcp:8080 \
    --source-ranges=0.0.0.0/0 \
    --target-tags=torchserve
}

InstallNvidiaDriverGCP() {
  curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output install_gpu_driver.py
  sudo python3 install_gpu_driver.py
  rm install_gpu_driver.py
  source ~/.bashrc
}

InstallNvidiaDriverGCP
AddFirewallRule