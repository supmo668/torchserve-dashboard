FROM awsdeeplearningteam/multi-model-server:latest-gpu

USER model-server
WORKDIR /home/model-server
COPY ./modelstore/* /home/model-server/model/

ENTRYPOINT multi-model-server
# CMD ["--start", "--model-store", "./model", "--models", "web_google-large-dino-20231011_T141010"]