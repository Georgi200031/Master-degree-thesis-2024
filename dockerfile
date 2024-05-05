# Use an official Python runtime as a parent image
FROM python:3.8

# Install necessary libraries for OpenCV and OpenGL
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx && \
    apt-get install python3-tk && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME fingerPrintRecognize

# Run app.py when the container launches
CMD ["python", "./main.py"]

