#Athena
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04 
#Zeus
#FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu18.04
#Styx 
#FROM nvidia/cuda:11.0-cudnn8-runtime-ubuntu18.04

ARG NAME=open-clip

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

ENV TZ=Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y --no-install-recommends \
	build-essential \
    ccache \
    cmake \
    pkg-config \
    python3 \
    python3-dev \
    python3-pip \
	libc6-dev \
    git \
	ffmpeg && \
	python3 -m pip install --upgrade --no-cache-dir pip setuptools wheel

# Uncomment this if you are working on a server with python version < 3.7
#RUN apt-get install -y --no-install-recommends python3.7  python3.7-venv python3.7-dev
#RUN update-alternatives --install /usr/bin/python python3 /usr/bin/python3.7 1
# Set python3.7 as the default python
#RUN update-alternatives --set python3 /usr/bin/python3.7
#RUN python -m pip install --upgrade --no-cache-dir pip setuptools wheel

WORKDIR /home/${NAME}

COPY . .

RUN python3 -m pip install --no-cache-dir . && \
    module_path="$(find /usr/local/lib -name "${NAME}" -type d)" && \
    rm -rf $module_path && \
    ln -sf $PWD/${NAME}/ $module_path && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/*


