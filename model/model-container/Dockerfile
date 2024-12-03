FROM python:3.12-slim

RUN apt-get update && \
    apt-get install -y wget && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -f -p /opt/conda && \
    rm miniconda.sh && \
    /opt/conda/bin/conda install -c conda-forge arviz matplotlib numpy pandas "pymc==5.16.2" xarray pytensor scipy && \
    /opt/conda/bin/conda clean -ya

WORKDIR /app

COPY . /app

EXPOSE 5000