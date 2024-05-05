# Gold price prediction

Brief description of your project.

## Prerequisites

- Docker installed on your machine.

## Installing Docker on Ubuntu

Follow the steps below to install Docker on Ubuntu:
   # Add Docker's official GPG key:
   sudo apt-get update
   sudo apt-get install ca-certificates curl
   sudo install -m 0755 -d /etc/apt/keyrings
   sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
   sudo chmod a+r /etc/apt/keyrings/docker.asc

   # Add the repository to Apt sources:
   echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
    $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
   sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
   sudo apt-get update
   sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

## Docker commands
Follow the step to work with Docker   

   1. Run Docker Service
        sudo service docker start
   2. Build docker file
	sudo docker build -t dockerfile .
   3. Run docker file
        sudo docker run -d -p 8080:80 thesis
   4. Run directly
	sudo docker run -it <container name>
   5. For open docker image after run use
	sudo docker logs <container id>
   6. For view all runing dockers 
	use this command sudo docker ps -a
   7. For remove docker 
	use this command sudo docker rm <container id>
   8. For stop docker container
	use this command sudo docker stop <container id>
