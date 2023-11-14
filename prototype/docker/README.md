# Deploying & Running the application

### Option 2: Run by docker-compose [recommended]
To start the applications, run docker-compose on the ```docker-compose_vision.yaml``` such as follows:
```
docker-compose --project-directory ./ -f deploy/docker-compose_vision.yaml up -d
```
or in 2-stage for whatever reason to not run the container:
```
docker-compose --project-directory ./ -f deploy/docker-compose_vision.yaml build
docker-compose --project-directory ./ -f deploy/docker-compose_vision.yaml run --service-ports -d --name ezout_vision ezout_vision
```
You may substitute the docker deployment configuration file with ```docker-compose_full.yaml``` to deploy all the applications. However, that script and other application is not tested at time of writing.

### Option 1: Run from dockerfile
For example, to just run vision app server docker, run the following from the repository home directory:
```
# build the image
docker build . -t ezout_vision -f EZOutDeploy\dockerfiles\vision.Dockerfile
```
and run the image with desired options, for example with:
```
# run the image (with application port)
docker run -dit -p 5000:5000 --name ezout_vision ezout_vision 
```

