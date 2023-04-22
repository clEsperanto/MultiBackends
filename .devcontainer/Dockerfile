FROM codespaces/codespaces-standard:ubuntu-18.04

# Install NVIDIA drivers and set up CUDA environment variables
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && \
    apt-get -y install --no-install-recommends gnupg2 curl ca-certificates && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - && \
    echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    apt-get update && \
    apt-get -y install --no-install-recommends cuda-toolkit-11-0 && \
    rm -rf /var/lib/apt/lists/*

# Set up OpenCL environment variables
ENV LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64:${LD_LIBRARY_PATH}
