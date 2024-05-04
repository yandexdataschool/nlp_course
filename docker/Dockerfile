FROM ubuntu:16.04
LABEL maintainer "NLP Team <no@ema.il>"


RUN echo "deb http://archive.ubuntu.com/ubuntu trusty-backports main restricted universe multiverse" >> /etc/apt/sources.list && \
    apt-get -qq update && \
    apt-get install -y cmake \
                       wget \
                       unzip \
                       git \
                       zlib1g-dev \
                       libjpeg-dev \
                       xvfb \
                       xorg-dev \
                       python-opengl \
                       python-dev \
                       python3-dev \
                       python-pip \
                       python3-pip \
                       graphviz \
                       gcc \
                       g++ && \
    ln -s /usr/bin/swig3.0 /usr/bin/swig

RUN pip install --upgrade pip==9.0.3 && \
    pip install --upgrade numpy scipy && \
    pip install --upgrade sklearn \
                           jupyter \
                           tqdm \
                           graphviz \
                           nltk \
                           gensim \
                           bokeh \
                           editdistance \
                           matplotlib \
                           seaborn && \
    pip install --upgrade  tensorflow \
                           http://download.pytorch.org/whl/cpu/torch-0.4.0-cp27-cp27mu-linux_x86_64.whl \
                           torchvision \
                           keras

RUN pip3 install --upgrade pip==9.0.3 && \
    pip3 install --upgrade numpy scipy && \
    pip3 install --upgrade sklearn \
                           jupyter \
                           tqdm \
                           graphviz \
                           nltk \
                           gensim \
                           bokeh \
                           editdistance \
                           matplotlib \
                           seaborn && \
    pip3 install --upgrade http://download.pytorch.org/whl/cpu/torch-0.4.0-cp35-cp35m-linux_x86_64.whl \
                           torchvision \
                           tensorflow \
                           keras && \                           
    python3 -m ipykernel.kernelspec


EXPOSE 8888
VOLUME /notebooks
WORKDIR /notebooks

COPY run_jupyter.sh /
CMD ["/run_jupyter.sh"]
