# Use an official NVIDIA runtime as a parent image
FROM nvidia/cuda:12.3.1-base-ubuntu22.04

# Install basics
RUN apt-get update && apt-get install -y python3-pip
RUN pip3 install --upgrade pip

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY server/* .

# Install any needed packages specified in requirements.txt
RUN pip3 install -r requirements.txt

# Run app.py when the container launches
CMD "python3 -m server.app"
