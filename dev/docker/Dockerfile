# syntax=docker/dockerfile:1

FROM stereolabs/zed:3.7-gl-devel-cuda11.4-ubuntu18.04

WORKDIR /home/lin/Projects/Human_Catch_Throw

RUN echo "deb [arch=amd64 trusted=yes] https://apt.prophesee.ai/dists/public/7l58osgr/ubuntu bionic essentials" >> /etc/apt/sources.list

RUN rm /etc/apt/sources.list.d/cuda.list

RUN apt update
RUN apt upgrade -y

RUN pip3 install --upgrade pip

RUN apt -y install libcanberra-gtk-module mesa-utils ffmpeg
RUN apt -y install libsm6 libxext6
RUN apt -y install python3-tk
RUN apt -y install python3.6-dev
RUN pip3 install "numpy==1.19.5" "opencv-python>=4.2.0.34" matplotlib jupyter pandas
# RUN pip3 install jupyter jupyterlab matplotlib "ipywidgets==7.6.5"
RUN apt -y install cmake libboost-program-options-dev libeigen3-dev
# RUN apt -y install metavision-sdk
RUN apt -y install metavision-essentials

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt



