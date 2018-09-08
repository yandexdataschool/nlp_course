FROM andrewosh/binder-base
MAINTAINER YSDA <jheuristic@yandex-team.ru>
USER root

RUN echo "deb http://archive.ubuntu.com/ubuntu trusty-backports main restricted universe multiverse" >> /etc/apt/sources.list
RUN apt-get -qq update

RUN apt-get install -y gcc-4.9 g++-4.9 libstdc++6 wget unzip
RUN apt-get install -y libopenblas-dev liblapack-dev libsdl2-dev libboost-all-dev graphviz
RUN apt-get install -y cmake zlib1g-dev libjpeg-dev 
RUN apt-get install -y xvfb libav-tools xorg-dev python-opengl python3-opengl
RUN apt-get -y install swig3.0
RUN ln -s /usr/bin/swig3.0 /usr/bin/swig


USER main
RUN pip install --upgrade pip==9.0.3
RUN pip install --upgrade --ignore-installed setuptools  #fix https://github.com/tensorflow/tensorflow/issues/622
RUN pip install --upgrade numpy scipy pandas sklearn tqdm joblib graphviz bokeh python-igraph
RUN pip install --upgrade nltk gensim editdistance 
RUN pip install --upgrade http://download.pytorch.org/whl/cpu/torch-0.4.0-cp27-cp27mu-linux_x86_64.whl 
RUN pip install --upgrade torchvision 




RUN /home/main/anaconda/envs/python3/bin/pip install --upgrade pip==9.0.3
RUN /home/main/anaconda/envs/python3/bin/pip install --upgrade --ignore-installed setuptools  #fix https://github.com/tensorflow/tensorflow/issues/622

# python3: fix `GLIBCXX_3.4.20' not found - conda's libgcc blocked system's gcc-4.9 and libstdc++6
RUN bash -c "conda update -y conda && source activate python3 && conda uninstall -y libgcc && source deactivate"

RUN /home/main/anaconda/envs/python3/bin/pip install --upgrade numpy scipy pandas sklearn tqdm joblib graphviz bokeh python-igraph
RUN /home/main/anaconda/envs/python3/bin/pip install --upgrade nltk gensim editdistance 
RUN /home/main/anaconda/envs/python3/bin/pip install --upgrade http://download.pytorch.org/whl/cpu/torch-0.4.0-cp35-cp35m-linux_x86_64.whl 
RUN /home/main/anaconda/envs/python3/bin/pip install --upgrade torchvision


#install TF after everything else not to break python3's pyglet with python2's tensorflow
RUN pip install --upgrade tensorflow keras
RUN /home/main/anaconda/envs/python3/bin/pip install --upgrade tensorflow keras
