FROM ghcr.io/kastnerrg/cse160-opencl:gpu-adreno

USER root

RUN apt-get update && \
    apt-get install -y libopencv-dev pkg-config && \
    rm -rf /var/lib/apt/lists/*

USER ubuntu