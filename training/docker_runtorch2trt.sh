#!/usr/bin/env bash

sudo systemctl restart docker
set -e

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
WORKSPACE="$(pwd):/workspace/"
USER_VOLUME1="/home/brian/models:/models"
USER_VOLUME2="/home/brian/projects:/projects"
CONTAINER_NAME="torch2trt"
CONTAINER_IMAGE="torch2trt:0.3"

echo "CONTAINER_IMAGE: $CONTAINER_IMAGE"
echo "CONTAINER_NAME: $CONTAINER_NAME"
echo "USER_VOLUME1:     $USER_VOLUME1"
echo "USER_VOLUME2:     $USER_VOLUME2"

# run the container

sudo docker run -it --rm --gpus all --ipc=host --network host --shm-size 8G\
    --name $CONTAINER_NAME -v $WORKSPACE -v $USER_VOLUME1 -v $USER_VOLUME2 $CONTAINER_IMAGE $USER_COMMAND
