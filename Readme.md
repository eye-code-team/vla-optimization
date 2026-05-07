# dynamic_layer_skipping_lora_prunning_snapflow

This repository contains the Docker-based setup and training scripts for the SmolVLA fine-tuning pipeline, including `finetune_dynamic_lora_prunning_snapflow.py` and helper automation scripts.

## Repository and Branch

- Repository: https://github.com/eye-code-team/vla-optimization
- Branch: `dynamic-layer-skipping`

## Docker Image

Use the published Docker image to run the code in a reproducible environment:

```bash
docker pull ghcr.io/eye-code-team/vla-optimization:latest
```

If the image is also published on Docker Hub, you can use:

```bash
docker pull docker.io/eye-code-team/vla-optimization:latest
```

## Prerequisites

- Docker Engine installed
- NVIDIA GPU drivers and NVIDIA Container Toolkit installed for GPU access
- Linux shell or WSL2 on Windows

## Quick Start

1. Clone the repository and checkout the branch:

```bash
git clone https://github.com/eye-code-team/vla-optimization.git
cd vla-optimization
git checkout dynamic-layer-skipping
```

2. Copy the environment template if needed:

```bash
cp .env.template .env
```

3. Prepare the host directories:

```bash
./setup.sh
```

This creates the standard host folders:

- `data/datasets`
- `data/checkpoints`
- `data/hf_cache`
- `outputs`
- `.env`

4. Pull the Docker image:

```bash
docker pull ghcr.io/eye-code-team/vla-optimization:latest
```

5. Start a container with mounted code, datasets, checkpoints, and outputs:

```bash
docker run --gpus all -it --user $(id -u):$(id -g) \
  --name smolvla_dev \
  --ipc=host \
  -v $PWD:/workspace/your_project \
  -v $PWD/data/datasets:/datasets \
  -v $PWD/data/checkpoints:/checkpoints \
  -v $PWD/outputs:/outputs \
  -w /workspace/your_project \
  ghcr.io/eye-code-team/vla-optimization:latest bash
```

6. Inside the container, run the training script:

```bash
python finetune_dynamic_lora_prunning_snapflow.py
```

or use the helper script:

```bash
./finetune.sh
```

## Recommended Workflow

- Keep source code in the repository mount (`/workspace/your_project`).
- Keep datasets in `/datasets` and checkpoints in `/checkpoints`.
- Keep output files in `/outputs` so they persist on the host.
- Do not bake large data files into the Docker image.

## Useful Commands

Start an interactive shell in the container:

```bash
./run.sh bash
```

Run the smoke test for the environment:

```bash
./test_setup.sh
```

Rebuild the image if you change Docker dependencies:

```bash
./run.sh rebuild
```

## Notes

- The Docker image should contain only the runtime environment and dependencies.
- Data, checkpoints, and outputs should be mounted from the host to keep the image portable.
- For a more detailed Docker workflow, see `README_docker_environment_guide.md`.
