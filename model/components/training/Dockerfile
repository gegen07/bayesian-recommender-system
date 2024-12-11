# Use an official Python runtime as a parent image
FROM python:3.12-slim as builder

# Install base utilities
RUN apt-get update \
    && apt-get install -y build-essential \
    && apt-get install -y wget \
    && apt-get install -y libarchive-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

RUN conda create -n pmf
RUN echo "source activate pmf" > ~/.bashrc
ENV PATH=/opt/conda/envs/pmf/bin:$PATH

RUN conda install -c conda-forge -y pandas fsspec gcsfs pymc==5.16.2 numpy matplotlib arviz xarray pytensor scipy
RUN conda clean --all --yes

FROM python:3.12-slim as final

ENV CONDA_DIR=/opt/conda
RUN mkdir -p $CONDA_DIR

COPY --from=builder /opt/conda /opt/conda

ENV PATH=$CONDA_DIR/bin:$PATH

WORKDIR /app

COPY . /app

ENV PATH=$CONDA_DIR/envs/pmf/bin:$PATH

EXPOSE 5000