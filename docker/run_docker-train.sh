docker build -t ezout-vision/train -f Dockerfile.train.ts-dashboard . --no-cache
docker run -dit --rm \
  -v ./datasets:/home/model-server/datasets \
  -v ./config:/home/model-server/config \
  -v ./checkpoint:/home/model-server/checkpoint \
  -v ./entrypoint:/usr/local/bin/ \
  -P 6006:6006 \
  --name ezout-vision-train ts-dashboard/train -I ./datasets -C datasets_11202023