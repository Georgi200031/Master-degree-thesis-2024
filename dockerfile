# Use an official Python runtime as a parent image
FROM python:3.10.12

# Install necessary libraries for OpenCV and OpenGL
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx python3-tk xvfb && \
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
ENV NAME stockpredictprice

# Start Xvfb and then run main.py when the container launches
CMD ["bash", "-c", "Xvfb :99 -screen 0 1024x768x16 & export DISPLAY=:99 && python ./main.py"]
