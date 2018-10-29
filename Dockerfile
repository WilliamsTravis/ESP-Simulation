FROM jupyter/scipy-notebook

MAINTAINER Max Joseph <maxwell.b.joseph@colorado.edu>

USER root

RUN apt-get update && \
    apt-get install -y libfreetype6-dev pkg-config libx11-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN conda config --remove channels defaults && \
  conda config --add channels conda-forge && \
  conda install --yes -c conda-forge \
    awscli \
    dash \
    dash-core-components \
    dash-html-components \
    flask-cache \
    pandas

RUN conda install --yes -c conda-forge flask-caching

# copy scripts into the container
COPY . /home/docker/

# specify that the location of script is the working directory
WORKDIR /home/docker

CMD ["python", "scripts/vis_app.py"]
