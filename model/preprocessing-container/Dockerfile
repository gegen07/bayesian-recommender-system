# Use an official Python runtime as a parent image
FROM python:3.12-slim

RUN apt-get update && \
    apt-get install -y wget && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -f -p /opt/conda && \
    rm miniconda.sh && \
    /opt/conda/bin/conda install -c conda-forge pandas "pymc==5.16.2" numpy && \
    /opt/conda/bin/conda clean -ya

WORKDIR /app

COPY . /app

EXPOSE 5000