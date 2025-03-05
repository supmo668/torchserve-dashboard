# EzOut TorchServe MMS containers

## Pre-req
Model artifacts should be available to be either mapped or copied into 
## Quickstart usage with docker
Use docker compose to create the containers
```
docker-compose -f dockercompose.ts-ezout.yml up -d
```
1. TorchServe (w/ dashboard, Prometheus/Grafana) 
  ```
  docker-compose -f dockercompose.vision-serve.yml up -d
  ```
  Assumes the model archiv and configuration to be in the following local directory which will be mapped to the docker container
  ```
  - ./modelstore:/home/model-server/model-store
  - ./config:/home/model-server/config
  - ./entrypoint:/usr/local/bin
  ```
  which will be used as :
  
    * model archiv storage 
    
    * configurations for torchserve & dasboards
    
    * entrypoint scripts for defining how the application is run
  
  (Reference The torchserve dashboard repo has been extended from [torchserve-dashboard](https://github.com/cceyda/torchserve-dashboard/tree/main))
  
2. Training
```
docker-compose -f dockercompose.vision-train.yml up -d
```
3. Hosting
* First expore the model via the script ```torch-archiv-export.sh``` by install `torch-model-archiver`:
  ```
  python -m pip install -r requirements.txt
  ```
[//]: # "All the references in this file should be actual links because this file would be used by docker hub. DO NOT use relative links or section tagging."

## Detail Dev Resources

### Design: Containerized Service 
The docker definitions are the application environment namely docker compose files:
* dockercompose.vision-serve.yml
* dockercompose.vision-train.yml
  ... etc

## Other references

1. [MMS from AWS](README-MMS.md)