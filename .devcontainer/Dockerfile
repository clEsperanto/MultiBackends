# Use the NVIDIA CUDA base image
FROM nvidia/cuda:11.4.2-devel-ubuntu20.04

# Set the environment variable to avoid tzdata interaction
ENV DEBIAN_FRONTEND=noninteractive

# Install CMake and g++ compiler
RUN apt-get update && apt-get install -y build-essential cmake

# Configure the timezone
RUN ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata
    
# Set the working directory to /workspace
WORKDIR /workspace
