FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS comfyui

ARG APT_MIRROR=mirrors.aliyun.com
ARG PIP_INDEX=https://mirrors.aliyun.com/pypi/simple
ARG TIMEZONE=Asia/Shanghai
ARG UID=1000
ARG GID=1000
ARG UNAME=ComfyUI

# Install 3rd party apps
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
RUN apt-get update && \
    apt-get install -y --no-install-recommends tzdata ffmpeg libsox-dev parallel aria2 git git-lfs && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y --no-install-recommends cmake build-essential curl wget git python-is-python3 python3-dev python3-pip vim && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade -i $PIP_INDEX pysocks 'requests[socks]' openpyxl

# Copy only requirements.txt initially to leverage Docker cache
WORKDIR /workspace
COPY requirements.txt /workspace/
RUN pip install --no-cache-dir --upgrade -i $PIP_INDEX --extra-index-url https://download.pytorch.org/whl/cu124 --default-timeout=100 -r requirements.txt


RUN pip install --no-cache-dir --upgrade -i $PIP_INDEX matplotlib==3.7.0

# Define a build-time argument for image type
ARG IMAGE_TYPE=full

# Conditional logic based on the IMAGE_TYPE argument
# Always copy the Docker directory, but only use it if IMAGE_TYPE is not "elite"
COPY ./Docker /workspace/Docker 
# elite 类型的镜像里面不包含额外的模型
RUN if [ "$IMAGE_TYPE" != "elite" ]; then \
        chmod +x /workspace/Docker/download.sh && \
        /workspace/Docker/download.sh && \
        python /workspace/Docker/download.py && \
        python -m nltk.downloader averaged_perceptron_tagger cmudict; \
    fi



# Copy the rest of the application
COPY . /workspace

EXPOSE 9871 9872 9873 9874 9880

CMD ["python", "webui.py"]
