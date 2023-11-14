# version: '3.8'
# Part of the implementation of this container is based on the Amazon SageMaker Apache MXNet container.
# https://github.com/aws/sagemaker-mxnet-container

# sagemaker environment variables
ARG SAGEMAKER_TRAINING_MODULE ezout_yolovision_framework_training.training:main
ARG SAGEMAKER_PROGRAM train.py 
ARG TENSORBOARD_PORT=8888

ARG REGION=us-east-2
# 1. BUILD stage: extend pre-built
# FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel as builder
FROM nvcr.io/nvidia/pytorch:22.11-py3 AS builder

# Defining some variables used at build time to install Python3
ARG PYTHON=python3
ARG PYTHON_PIP=python3-pip
ARG PIP=pip3

ENV TZ=US/New_York
ENV DEBIAN_FRONTEND noninteractive
ENV PYTHON_VERSION=3.9

# Install some handful libraries like curl, wget, git, build-essential, zlib
RUN echo 'tzdata tzdata/Areas select Europe' | debconf-set-selections && echo 'tzdata tzdata/Zones/Europe select Paris' | debconf-set-selections 

RUN DEBIAN_FRONTEND="noninteractive" apt-get update && \
    apt-get install -y --no-install-recommends apt-utils software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        python3-dev \
        build-essential \
        ca-certificates \
        curl \
        wget \
        git \
        libopencv-dev \
        openssh-client \
        openssh-server \
        vim \
        ca-certificates \
        openjdk-8-jdk-headless \
        && rm -rf /var/lib/apt/lists/* 

FROM builder AS runner
RUN useradd -m model-server
WORKDIR /
ENV PATH="/home/model-server/.local/bin:${PATH}"
RUN python -m pip install -U pip && pip install --no-cache --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple -U ezout-yolovision multi-model-server

FROM runner AS sg_yolonas_base

USER root
WORKDIR /home/model-server
# Setting some environment variables.

ENV PATH="/opt/ml/code:${PATH}"
ARG TENSORBOARD_PORT
ARG MODEL_PATH
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${LIB_DIR}" \
    PYTHONIOENCODING=UTF-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    AWS_REGION_NAME=us-east-2
ENV CODE_SRC=mms_ezout \
    TEMP=/home/model-server/tmp \
    LIB_DIR=/usr/local/lib \
    MODEL_PATH=${MODEL_PATH}

EXPOSE 8080 8081
ENV PORT 8080

# Copy the default custom service file to handle incoming data and inference requests
COPY ${CODE_SRC}/dockerd-entrypoint.py $LIB_DIR/
COPY ${CODE_SRC}/model_handler.py /opt/ml/code/
COPY ${CODE_SRC}/config.properties /home/model-server/
COPY dockerd-entrypoint.sh $LIB_DIR/

COPY ${MODEL_PATH}/* /opt/ml/model/


RUN mkdir -p ${TEMP} && chmod +x $LIB_DIR/dockerd-entrypoint.sh \
    && chown -R model-server ./
  
FROM sg_yolonas_base AS serve
# FROM pytorch/torchserve:latest-gpu

USER model-server
# Archiv model
# RUN git clone https://github.com/pytorch/serve.git && cd serve/model-archiver && pip install .
# torch-model-archiver -f \
#   --model-name="ezout_vision-$(basename $(dirname $MODEL))" \
#   --version=1.0 \
#   --serialized-file=$MODEL_PATH \
#   --handler=$HANDLER \
#   --extra-files $(dirname $MODEL)/data.yaml \
#   --export-path=$(dirname $MODEL)

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# USER model-server:model-server
ENTRYPOINT ["/usr/local/lib/dockerd-entrypoint.sh"]
CMD ["serve"]

LABEL maintainer="EzOut Vision Team, mmym.ezout@gmail.com, info@ezout.store"