#!/bin/bash
if [ $1 = "run" ]
then
    echo "run model_make"
    # docker run -d -it --gpus all -v $PWD:/home -w /home --name model_make tensorflow/tensorflow:2.5.0-gpu bash
    docker run -d -it --gpus all -v $PWD:/home -w /home -e DISPLAY=$DISPLAY --env QT_X11_NO_MITSSHM=1 --name model_make linakim/tf2:1.0 /bin/bash
    # docker run -d -it --gpus all --privileged -v $PWD:/home -v /dev/video0:/dev/video0 -v /tmp/.X11-unix:/tmp/.X11-unix -w /home -e DISPLAY=$DISPLAY --env QT_X11_NO_MITSSHM=1 --name model_make linakim/tf2:1.0 /bin/bash
elif [ $1 = "start" ]
then
    echo "start model_make"
    docker start model_make
elif [ $1 = "stop" ]
then
    echo "stop model_make"
    docker stop model_make
else
    echo "Usage: run.sh [run|start|stop]"
fi