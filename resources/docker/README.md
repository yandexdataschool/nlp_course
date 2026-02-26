To simplify installation process, you can deploy a container (~virtual machine) with all dependencies pre-installed.

_tl;dr [dockerhub url](https://hub.docker.com/r/justheuristic/nlp_course/)_

## Install Docker

We recommend you to use either native docker (recommended for linux) or kitematic(recommended for windows).
* Installing [kitematic](https://kitematic.com/), a simple interface to docker (all platforms)
* Pure docker: Guide for [windows](https://docs.docker.com/docker-for-windows/), [linux](https://docs.docker.com/engine/installation/), or [macOS](https://docs.docker.com/docker-for-mac/).

Below are the instructions for both approaches.

## Kitematic
Find justheuristic/nlp_course in the search menu. Download and launch the container.

Click on "web preview" screen in the top-right __or__ go to settings, ports and find at which port your jupyter is located, usually 32***.

## Native
`docker run -it -v <local_dir>:/notebooks -p <local_port>:8888 justheuristic/nlp_course sh ../run_jupyter.sh`

For example,
`docker run -it -v /Users/mittov/Documents/shad/semester4/ -p 8888:8888 justheuristic/nlp_course sh ../run_jupyter.sh`


Then you can access your jupyter in a browser at `localhost:<local_port>`. 

If it asks you to enter token, you will find one in container logs, e.g. `localhost:8888/?token=ad1a5a0aab43efb47a9a805388fcf508d0b5f84a16e4542b&token=ad1a5a0aab43efb47a9a805388fcf508d0b5f84a16e4542b`.

## Manual
Build container

`$ docker build -t nlp .`


Run it

`$ docker run --rm -it -v <local_dir>:/notebooks -p <local_port>:8888 nlp`

examples:

`$ docker run --rm -it -v /Users/mittov/Documents/shad/semester4/:/notebooks -p 8888:8888 nlp`

Copy the token from console and run
http://localhost:8888/ (if it asks for token, see "native")
