ARG CONTEXT="runtime"
FROM registry.k.avito.ru/avito/service-python/cuda/${CONTEXT}:3.8.6-cuda10.2.89-cudnn7.6.5.32
ARG CONTEXT="runtime"

COPY ./requirements.txt $PROJECT_ROOT/

RUN apt-get update -y && \
    apt-get install ssh -y && \
    apt-get install git -y && \
    apt-get install git-lfs -y && \
    git lfs install

RUN python3 -m pip install -r requirements.txt

COPY ./jupyter_notebook_config.py /root/.jupyter/

COPY . $PROJECT_ROOT

RUN echo $USER && \
    apt-get install openssh-client

RUN apt-get clean && rm -rf /var/lib/apt/lists/*
