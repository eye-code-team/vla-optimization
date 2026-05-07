# Docker Environment Guide

## Docker Workflow Philosophy

Use the Docker image to provide a consistent working environment across different PCs and servers. The image should contain only the required packages and dependencies, while source code, datasets, checkpoints, and outputs should stay outside the image and be mounted at runtime. Repeated package installations or setup commands should be added to the `Dockerfile` to keep the environment reproducible.

This README describes how to build the Docker image, launch a Docker container, attach to a running container, package the Docker image, transfer it to another machine, and run it on the new machine.

## Fast Start for Handoff (Recommended)

Use this workflow when handing the project to a new user.

### Prerequisites

- Docker Engine + Docker Compose plugin
- NVIDIA driver + NVIDIA Container Toolkit (for GPU)
- Linux shell (or WSL2 on Windows)

### 1) Initialize project directories and environment file

```bash
./setup.sh
```

This creates:

```bash
data/datasets
data/checkpoints
data/hf_cache
outputs
.env
```

### 2) Build image and boot container with checks

```bash
./quick_start.sh
```

This will:

- build the Docker image
- start container `smolvla_dev`
- check GPU visibility (`nvidia-smi`)
- check Python + torch import

### Docker image link

If you publish a prebuilt image, pull it with:

```bash
docker pull ghcr.io/eye-code-team/vla-optimization:latest
```

Or use the Docker Hub equivalent if published there:

```bash
docker pull docker.io/eye-code-team/vla-optimization:latest
```

### 3) Enter container shell

```bash
./run.sh bash
```

### 4) Run the finetune pipeline

```bash
./finetune.sh
```

The script runs:

```bash
python finetune_dynamic_lora_prunning_snapflow.py
```

with Docker-mounted outputs at:

```bash
./outputs
```

### 5) Useful commands

```bash
./run.sh up       # start in background
./run.sh logs     # follow logs
./run.sh down     # stop and remove container
./run.sh rebuild  # rebuild image without cache
./test_setup.sh   # full environment smoke test
```

### 6) Optional dataset download helper

```bash
./download_sample_dataset.sh lerobot/svla_so100_pickplace
```

By default this downloads to `/datasets/<repo_name>` inside container, mapped to host `./data/datasets`.

## Notes for Current Docker Setup

- Image includes environment and pre-cached base model `lerobot/smolvla_base`.
- Source code, datasets, checkpoints, and outputs stay outside image and are mounted at runtime.
- You can tune mount paths and names in `.env`.

---

## 1. Build the Docker Image

Run the following command in the directory that contains the `Dockerfile`:

```bash
docker build -t ${IMAGE_NAME:-your-project-env} .
```

After the build finishes, check that the image exists:

```bash
docker images | grep ${IMAGE_NAME:-your-project-env}
```

The image name should be:

```bash
${IMAGE_NAME:-your-project-env}:latest
```

You can replace `your-project-env` with the actual name of your project environment.

---

## 2. Set the Project Root Path

Before launching the container, define a project root path on the host machine. This avoids hard-coding personal paths in the README.

```bash
export PROJECT_ROOT=/path/to/your/project_root
```

Example:

```bash
export PROJECT_ROOT=/data/project_root
```

The expected directory structure is:

```bash
${PROJECT_ROOT}/datasets
${PROJECT_ROOT}/checkpoints
${PROJECT_ROOT}/outputs
```

Optionally, define reusable names for the Docker image and container:

```bash
export IMAGE_NAME=your-project-env
export CONTAINER_NAME=your_project_dev
```

---

## 3. Launch a Docker Container

### Option 1: Run with `${PROJECT_ROOT}`

```bash
docker run --gpus all -it --user $(id -u):$(id -g) \
  --name ${CONTAINER_NAME:-your_project_dev} \
  --ipc=host \
  -v $PWD:/workspace/your_project \
  -v ${PROJECT_ROOT}/datasets:/datasets \
  -v ${PROJECT_ROOT}/checkpoints:/checkpoints \
  -v ${PROJECT_ROOT}/outputs:/outputs \
  -w /workspace/your_project \
  ${IMAGE_NAME:-your-project-env} bash
```

Explanation of the main options:

```bash
--gpus all
```

Allows the container to use all available NVIDIA GPUs.

```bash
--user $(id -u):$(id -g)
```

Runs the container using the current host user ID and group ID. This helps avoid permission issues where files created inside the container are owned by `root`.

```bash
--name your_project_dev
```

Sets the container name to `your_project_dev`.

```bash
--ipc=host
```

Uses the host IPC namespace. This is useful for PyTorch training and dataloaders that use shared memory.

```bash
-v $PWD:/workspace/your_project
```

Mounts the current code directory from the host machine into the container at `/workspace/your_project`.

```bash
-v ${PROJECT_ROOT}/datasets:/datasets
-v ${PROJECT_ROOT}/checkpoints:/checkpoints
-v ${PROJECT_ROOT}/outputs:/outputs
```

Mounts the dataset, checkpoint, and output directories from the host machine into the container.

```bash
-w /workspace/your_project
```

Sets the default working directory inside the container.

---

### Option 2: Run with another container name

```bash
docker run --gpus all -it \
  --name your_project_dev2 \
  --ipc=host \
  -v $PWD:/workspace/your_project \
  -v ${PROJECT_ROOT}/datasets:/datasets \
  -v ${PROJECT_ROOT}/checkpoints:/checkpoints \
  -v ${PROJECT_ROOT}/outputs:/outputs \
  -w /workspace/your_project \
  ${IMAGE_NAME:-your-project-env}:latest bash
```

This container is named:

```bash
your_project_dev2
```

---

## 4. Attach to a Running Container

If the container is already running, open a new terminal and attach to it:

```bash
docker exec -it ${CONTAINER_NAME:-your_project_dev} bash
```

Or attach to the second container:

```bash
docker exec -it your_project_dev2 bash
```

Check running containers:

```bash
docker ps
```

Check all containers, including stopped containers:

```bash
docker ps -a
```

---

## 5. Stop and Remove a Container

Stop a running container:

```bash
docker stop ${CONTAINER_NAME:-your_project_dev}
```

Remove the container:

```bash
docker rm ${CONTAINER_NAME:-your_project_dev}
```

If you want to recreate a container with the same name:

```bash
docker stop ${CONTAINER_NAME:-your_project_dev}
docker rm ${CONTAINER_NAME:-your_project_dev}
```

Then run the `docker run` command again.

---

## 6. Package the Docker Image

After building the image, save it as a `.tar` file:

```bash
docker save -o your-project-env.tar ${IMAGE_NAME:-your-project-env}:latest
```

Compress the `.tar` file:

```bash
gzip your-project-env.tar
```

The compressed file will be:

```bash
your-project-env.tar.gz
```

Check the file size:

```bash
ls -lh your-project-env.tar.gz
```

---

## 7. Transfer the Docker Image to Another Machine

Transfer the following file to the new machine:

```bash
your-project-env.tar.gz
```

You can use `scp`, `rsync`, `croc`, an external drive, or Google Drive.

Example using `scp`:

```bash
scp your-project-env.tar.gz user@new_machine:/path/to/destination/
```

Example using `rsync`:

```bash
rsync -avh --progress your-project-env.tar.gz user@new_machine:/path/to/destination/
```

---

## 8. Load the Docker Image on the New Machine

On the new machine, decompress the image file:

```bash
gunzip your-project-env.tar.gz
```

This will produce:

```bash
your-project-env.tar
```

Load the image into Docker:

```bash
docker load -i your-project-env.tar
```

Verify that the image was loaded successfully:

```bash
docker images | grep ${IMAGE_NAME:-your-project-env}
```

If `your-project-env` appears in the image list, the image has been loaded successfully.

---

## 9. Run the Container on the New Machine

Example: run the container on a new machine using the dataset directory under:

```bash
${PROJECT_ROOT}/datasets
```

Command:

```bash
docker run --gpus all -it --rm \
  --name ${CONTAINER_NAME:-your_project_dev} \
  --ipc=host \
  -v $PWD:/workspace/your_project \
  -v ${PROJECT_ROOT}/datasets:/datasets \
  -w /workspace/your_project \
  ${IMAGE_NAME:-your-project-env} bash
```

Explanation:

```bash
--rm
```

Automatically removes the container after it exits. This is useful for temporary runs.

If you want to keep the container and attach to it later, remove `--rm`:

```bash
docker run --gpus all -it \
  --name ${CONTAINER_NAME:-your_project_dev} \
  --ipc=host \
  -v $PWD:/workspace/your_project \
  -v ${PROJECT_ROOT}/datasets:/datasets \
  -w /workspace/your_project \
  ${IMAGE_NAME:-your-project-env} bash
```

---

## 10. Check GPU Access Inside the Container

After entering the container, check whether the GPU is visible:

```bash
nvidia-smi
```

If `nvidia-smi` shows the GPU information, the container can access the GPU.

You can also check PyTorch CUDA availability:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

Expected output:

```bash
True
```

and the number of available GPUs should be greater than 0.

---

## 11. Recommended Host Directory Structure

It is recommended to separate code, datasets, checkpoints, and outputs:

```bash
project_root/
├── datasets/
├── checkpoints/
├── outputs/
└── External/
    └── your-project/
        └── code/
```

For example, go to the code directory:

```bash
cd /path/to/your-project
```

Then launch the container:

```bash
docker run --gpus all -it --user $(id -u):$(id -g) \
  --name ${CONTAINER_NAME:-your_project_dev} \
  --ipc=host \
  -v $PWD:/workspace/your_project \
  -v ${PROJECT_ROOT}/datasets:/datasets \
  -v ${PROJECT_ROOT}/checkpoints:/checkpoints \
  -v ${PROJECT_ROOT}/outputs:/outputs \
  -w /workspace/your_project \
  ${IMAGE_NAME:-your-project-env} bash
```

Inside the container, the paths will be:

```bash
/workspace/your_project      # code
/datasets            # datasets
/checkpoints         # checkpoints
/outputs             # outputs
```

---

## 12. Common Issues

### Container name already exists

If you see an error like:

```bash
Conflict. The container name "/your_project_dev" is already in use
```

Remove the old container:

```bash
docker rm ${CONTAINER_NAME:-your_project_dev}
```

If the container is still running, stop it first:

```bash
docker stop ${CONTAINER_NAME:-your_project_dev}
docker rm ${CONTAINER_NAME:-your_project_dev}
```

---

### GPU is not visible inside the container

First, check whether the host machine can see the GPU:

```bash
nvidia-smi
```

If the host can see the GPU but the container cannot, test NVIDIA Docker support:

```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

If this command fails, the NVIDIA Container Toolkit may need to be installed or fixed.

---

### Permission issues when writing files

If files created inside the container are owned by `root`, run the container with the host user ID and group ID:

```bash
--user $(id -u):$(id -g)
```

Check the current user ID and group ID on the host:

```bash
id
```

Example output:

```bash
uid=1015(minh) gid=100(users)
```

Then use:

```bash
--user $(id -u):$(id -g)
```

---

## 13. Typical Workflow

### Build the image

```bash
docker build -t ${IMAGE_NAME:-your-project-env} .
```

### Run the container

```bash
docker run --gpus all -it --user $(id -u):$(id -g) \
  --name ${CONTAINER_NAME:-your_project_dev} \
  --ipc=host \
  -v $PWD:/workspace/your_project \
  -v ${PROJECT_ROOT}/datasets:/datasets \
  -v ${PROJECT_ROOT}/checkpoints:/checkpoints \
  -v ${PROJECT_ROOT}/outputs:/outputs \
  -w /workspace/your_project \
  ${IMAGE_NAME:-your-project-env} bash
```

### Attach to the container

```bash
docker exec -it ${CONTAINER_NAME:-your_project_dev} bash
```

### Save the Docker image

```bash
docker save -o your-project-env.tar ${IMAGE_NAME:-your-project-env}:latest
```

### Compress the image

```bash
gzip your-project-env.tar
```

### Load the image on a new machine

```bash
gunzip your-project-env.tar.gz
docker load -i your-project-env.tar
```

### Run the container on the new machine

```bash
docker run --gpus all -it \
  --name ${CONTAINER_NAME:-your_project_dev} \
  --ipc=host \
  -v $PWD:/workspace/your_project \
  -v ${PROJECT_ROOT}/datasets:/datasets \
  -w /workspace/your_project \
  ${IMAGE_NAME:-your-project-env} bash
```
