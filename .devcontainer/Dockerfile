# Use the nvidia/cuda base image
FROM nvidia/cuda:11.0-base

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cuda-command-line-tools-11-0 \
    libcudnn8=8.0.4.30-1+cuda11.0 \
    libcudnn8-dev=8.0.4.30-1+cuda11.0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/root/miniconda3/bin:$PATH
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    && rm -rf /var/lib/apt/lists/*
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
    && /bin/bash ~/miniconda.sh -b -p ~/miniconda3 \
    && rm ~/miniconda.sh
RUN conda init
RUN echo "conda activate base" >> ~/.bashrc

# Create a new conda environment and activate it
RUN conda create -n gpu_env python=3.8
SHELL ["conda", "run", "-n", "gpu_env", "/bin/bash", "-c"]
RUN echo "conda activate gpu_env" >> ~/.bashrc

# Install TensorFlow with GPU support
RUN conda install -y tensorflow-gpu

# Install PyTorch with GPU support
RUN conda install -y pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch

# Set the default command to run when the container starts
CMD ["/bin/bash"]