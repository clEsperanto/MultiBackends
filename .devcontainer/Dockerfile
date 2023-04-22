# Start with the CUDA base image
FROM nvidia/cuda:11.5.0-devel-ubuntu20.04

RUN apt-get update && \
    apt-get install -y build-essential cmake
    
RUN apt-get install -y ocl-icd-opencl-dev clhpp-headers

WORKDIR /app

COPY . /app

