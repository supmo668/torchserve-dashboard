# Debugging
## TorchServe Command

## Debugging with GPU container
* Debugging in a TorchServe environment
  * Start GPU container
  ```
  docker run --rm -itd --gpus all --name ts -p 127.0.0.1:8080:8080 -p 127.0.0.1:8081:8081 -p 127.0.0.1:8082:8082 -p 127.0.0.1:7070:7070 -p 127.0.0.1:7071:7071 -v %cd%\modelstore:/home/model-server/model-store pytorch/torchserve:latest-gpu
  ## Add packages 
  python -m pip install -U pip && pip install --no-cache --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple -U ezout-yolovision multi-model-server
  ```
  * Within the container, you may need to reboot to 
  ```
  torchserve --stop
  torchserve --start  --model-store=/home/model-server/model-store --models ALL
  ```
* (windows)
  *  Mount modelstore directly and serve locally
```
docker run -rm -it -p 8000:8080 -p 8001:8081 --name torchserve -e NVIDIA_VISIBLE_DEVICES=0 -e MMS_CONFIG_FILE=config.properties -v %cd%\modelstore:/home/model-server/model-store pytorch/torchserve:latest-gpu torchserve --start  --model-store=/home/model-server/model-store --models ALL
```

**
## Reference Commands
Unmodified document commands
```
docker run --rm -it --gpus all -p 127.0.0.1:8080:8080 -p 127.0.0.1:8081:8081 -p 127.0.0.1:8082:8082 -p 127.0.0.1:7070:7070 -p 127.0.0.1:7071:7071 pytorch/torchserve:latest-gpu
```

## Commands explanation
This is the container run command
```
torchserve --start  --model-store=/home/model-server/model-store --models ALL
```