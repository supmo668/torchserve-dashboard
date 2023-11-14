Help() {
  echo "build and run local docker. exception to use shared GPU using local nvidia runtime"
}

# PACKAGE_NAME=$(grep "name=" ../package/setup.py | cut -d'"' -f 2)
# VERSION=$(grep "version=" ../package/setup.py | cut -d'"' -f 2)

docker build -t ezout-vision/mms-ezout \
  -f Dockerfile.ezout-vision.gpu . --no-cache
  # --build-arg MODEL="./modelstore" \
  # --build-arg PACKAGE_NAME=$PACKAGE_NAME-$VERSION.tar.gz \
  
docker run -it -p 8080:8080 -p 8081:8081 --privileged \
  -e NVIDIA_VISIBLE_DEVICES=0  \
  -e MMS_CONFIG_FILE=config.properties \
  --name ezout-vision --rm \
  ezout-vision/yolonas-mms-deploy
  
