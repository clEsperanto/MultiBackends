FROM ubuntu:18.04

RUN apt-get update && apt-get install -y \
    apt-transport-https \
    build-essential \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    wget

# Install NVIDIA CUDA Toolkit
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin && \
    mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
    wget https://developer.download.nvidia.com/compute/cuda/11.4.2/local_installers/cuda-repo-ubuntu1804-11-4-local_11.4.2-470.57.02-1_amd64.deb && \
    dpkg -i cuda-repo-ubuntu1804-11-4-local_11.4.2-470.57.02-1_amd64.deb && \
    apt-key add /var/cuda-repo-ubuntu1804-11-4-local/7fa2af80.pub && \
    apt-get update && apt-get -y install cuda

# Install OpenCL
RUN apt-get install -y ocl-icd-opencl-dev

# Install OpenCL-CLHPP
RUN apt-get install -y clhpp-headers

# Set environment variables
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

