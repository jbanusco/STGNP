#!/bin/sh

# Ensure user provides "local" or "remote"
if [ -z "$1" ]; then
    echo "Usage: $0 [local|remote]"
    exit 1
fi

# Docker image name & version
IMAGE_NAME="multiplex"
VERSION="0.0"

# Local Docker registry
LOCAL_REGISTRY="localhost:5000"

# Remote Docker registry (Docker Hub)
DOCKER_HUB_USER="jbanusco"
REMOTE_REGISTRY="docker.io/${DOCKER_HUB_USER}"

# Singularity images path
SINGULARITY_IMGS="/media/jaume/DATA/Data/SingularityImages"

# Function to build and tag the Docker image
build_image() {
    echo "Building Docker image..."
    sudo docker build -f Dockerfile.cuda -t ${LOCAL_REGISTRY}/${IMAGE_NAME}:${VERSION} .
    sudo docker tag ${LOCAL_REGISTRY}/${IMAGE_NAME}:${VERSION} ${REMOTE_REGISTRY}/${IMAGE_NAME}:${VERSION}
}

# Function to push to local registry
push_local() {
    echo "Pushing to LOCAL registry: ${LOCAL_REGISTRY}/${IMAGE_NAME}:${VERSION}"
    sudo docker push ${LOCAL_REGISTRY}/${IMAGE_NAME}:${VERSION}
}

# Function to push to remote registry
push_remote() {
    echo "Pushing to REMOTE registry: ${REMOTE_REGISTRY}/${IMAGE_NAME}:${VERSION}"
    sudo docker login
    sudo docker push ${REMOTE_REGISTRY}/${IMAGE_NAME}:${VERSION}
}

# Function to convert image to Singularity SIF
convert_singularity() {
    echo "Converting Docker image to Singularity SIF..."
    rm -rfd ${SINGULARITY_IMGS}/${IMAGE_NAME}_${VERSION}.sif
    cd ${SINGULARITY_IMGS}

    if [ "$1" = "local" ]; then
        echo "Pulling from LOCAL registry for Singularity..."
        SINGULARITY_NOHTTPS=1 singularity pull docker://${LOCAL_REGISTRY}/${IMAGE_NAME}:${VERSION}
    else
        echo "Pulling from REMOTE registry for Singularity..."
        singularity pull docker://${REMOTE_REGISTRY}/${IMAGE_NAME}:${VERSION}
    fi
}

# Execute based on user choice
build_image
if [ "$1" = "local" ]; then
    push_local
    convert_singularity local
elif [ "$1" = "remote" ]; then
    push_remote
    convert_singularity remote
else
    echo "Error: Invalid option '$1'. Use 'local' or 'remote'."
    exit 1
fi

echo "Done!"
