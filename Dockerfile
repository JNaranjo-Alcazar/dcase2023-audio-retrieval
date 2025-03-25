# Use the official Python 3.9 image as the base image
FROM python:3.9-slim

### GPU support ###
# Use a CUDA-enabled base image (e.g., nvidia/cuda for GPU support)
#FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

# Install Python 3.9
#RUN apt-get update && apt-get install -y python3.9 python3-pip


# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch 1.13.0
RUN pip install --no-cache-dir torch==1.13.0

# Instal ray
RUN pip install ray[tune]

# Download nltk stopwords from python terminal
RUN python -c "import nltk; nltk.download('stopwords')"

### GPU support ###
# Install PyTorch 1.13.0 with CUDA 11.7 support
#RUN pip install --no-cache-dir torch==1.13.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117


# Copy the rest of the application code into the container
COPY . .