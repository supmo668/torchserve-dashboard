docker build -t ts-dashboard Dockerfile.ts-dashboard -f Dockerfile.ts-dashboard .
docker run -dit --rm -p 8105:8105 -p "8080-8082:8080-8082" \
  -v ./modelstore:/home/model-server/model-store \
  -v ./config:/home/model-server/config \
  -v ./entrypoint:/usr/local/bin/ ts-dashboard \
  --name ezout-vision ts-dashboard ts-dashboard 