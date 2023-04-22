# Start with the CUDA base image
FROM nvidia/cuda:11.5.0-devel-ubuntu20.04

# Install OpenCL and OpenCL-CLHPP
RUN apt-get update && \
    apt-get install -y ocl-icd-opencl-dev clhpp-headers

# Install CMake and a C++ compiler
RUN apt-get install -y cmake g++

# Set the working directory to /app
WORKDIR /app

# Copy the contents of the local directory to /app
COPY . /app

