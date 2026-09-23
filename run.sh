#!/bin/bash

# =============================================================================
# run.sh
# Container IsaacLab con:
# - cache/log ufficiali Isaac Sim (persistenza, evita errori CUDA/cusparseLt)
# - mount di source/ (codice custom, task presenti e futuri)
# - checkpoint del controllore di velocità reindirizzati nella host
# - installazione automatica (pip install -e) del pacchetto ad ogni avvio
# =============================================================================

IMAGE_NAME="nvcr.io/nvidia/isaac-lab:2.3.2"   # verifica con: docker images | grep isaac-lab
CONTAINER_NAME="isaac-lab-base_FM"

HOST_ISAACLAB=~/Project_Favuzzi_Mutasci
VEL_CKPT_DIR="$HOST_ISAACLAB/vel_controller_checkpoints"
POS_CKPT_DIR="$HOST_ISAACLAB/pos_controller_checkpoints"

mkdir -p "$VEL_CKPT_DIR"
mkdir -p "$POS_CKPT_DIR"
mkdir -p ~/docker/isaac-sim/cache/kit
mkdir -p ~/docker/isaac-sim/cache/ov
mkdir -p ~/docker/isaac-sim/cache/pip
mkdir -p ~/docker/isaac-sim/cache/glcache
mkdir -p ~/docker/isaac-sim/cache/computecache
mkdir -p ~/docker/isaac-sim/logs
mkdir -p ~/docker/isaac-sim/data
mkdir -p ~/docker/isaac-sim/documents

xhost +local:docker

# --- comando di setup + shell interattiva, eseguito DENTRO il container ---
STARTUP_CMD='
cd /workspace/project_workspace/Project_Favuzzi_Mutasci
/workspace/isaaclab/isaaclab.sh -p -m pip install -e source/Project_Favuzzi_Mutasci
exec bash
'

docker run --name "$CONTAINER_NAME" --entrypoint bash -it --rm \
    --gpus all \
    --network=host \
    --privileged \
    -e ACCEPT_EULA=Y \
    -e PRIVACY_CONSENT=Y \
    -e DISPLAY \
    -v "$HOME/.Xauthority":/root/.Xauthority \
    -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
    -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
    -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
    -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
    -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
    -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
    -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
    -v ~/docker/isaac-sim/documents:/root/Documents:rw \
    -v "$HOST_ISAACLAB/isaaclab_ws":/workspace/project_workspace \
    -v "$VEL_CKPT_DIR":/workspace/isaaclab/logs/skrl/vel_controller/ \
    -v "$POS_CKPT_DIR":/workspace/isaaclab/logs/skrl/pos_controller/ \
    "$IMAGE_NAME" -c "$STARTUP_CMD"