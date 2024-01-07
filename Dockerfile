# Use an official NVIDIA runtime as a parent image
FROM 12.3.1-base-ubuntu22.04

# Install any needed packages specified in requirements.txt
RUN apt-get update && apt-get install -y python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install tensorflow flask

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY server/app.py .

# Run app.py when the container launches
CMD ["python3", "app.py"]
