# Use the NVIDIA CUDA base image
FROM nvidia/cuda:11.4.2-devel-ubuntu20.04

# Install CMake and g++ compiler
RUN apt-get update && apt-get install -y build-essential cmake

# Configure the timezone
RUN echo "tzdata tzdata/Areas select America" > /tmp/tzdata.txt \
    && echo "tzdata tzdata/Zones/America select New_York" >> /tmp/tzdata.txt \
    && debconf-set-selections /tmp/tzdata.txt \
    && rm /etc/localtime \
    && ln -s /usr/share/zoneinfo/America/New_York /etc/localtime \
    && dpkg-reconfigure -f noninteractive tzdata
    
# Set the working directory to /workspace
WORKDIR /workspace
