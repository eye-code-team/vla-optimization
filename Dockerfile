ARG CUDA_VERSION=12.4.1
ARG UBUNTU_VERSION=22.04
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu${UBUNTU_VERSION}

ARG PYTHON_VERSION=3.12
ARG USER_NAME=vla_user
ARG USER_UID=1000
ARG USER_GID=1000

ENV DEBIAN_FRONTEND=noninteractive \
    MUJOCO_GL=egl \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    git \
    curl \
    ffmpeg \
    libglib2.0-0 \
    libgl1 \
    libegl1 \
    libusb-1.0-0-dev \
    speech-dispatcher \
    libgeos-dev \
    portaudio19-dev \
    cmake \
    pkg-config \
    ninja-build \
    ca-certificates \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-venv \
        python${PYTHON_VERSION}-dev \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv \
    && groupadd --gid ${USER_GID} ${USER_NAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} --create-home --shell /bin/bash ${USER_NAME} \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/lerobot

ENV HOME=/home/${USER_NAME} \
    HF_HOME=/home/${USER_NAME}/.cache/huggingface \
    HF_LEROBOT_HOME=/home/${USER_NAME}/.cache/huggingface/lerobot \
    TORCH_HOME=/home/${USER_NAME}/.cache/torch \
    TRITON_CACHE_DIR=/home/${USER_NAME}/.cache/triton \
    PATH=/opt/lerobot/.venv/bin:$PATH

RUN mkdir -p ${HF_HOME} ${HF_LEROBOT_HOME} ${TORCH_HOME} ${TRITON_CACHE_DIR}

COPY --chown=${USER_NAME}:${USER_NAME} lerobot/setup.py lerobot/pyproject.toml lerobot/uv.lock lerobot/README.md lerobot/MANIFEST.in ./
COPY --chown=${USER_NAME}:${USER_NAME} lerobot/src/ src/

RUN rm -rf /opt/lerobot/src/lerobot.egg-info \
    && chmod -R a+rwX /opt/lerobot \
    && chown -R ${USER_NAME}:${USER_NAME} /opt/lerobot /home/${USER_NAME}/.cache

USER ${USER_NAME}

RUN uv venv --python python${PYTHON_VERSION} \
    && uv sync --locked --extra all --no-dev

RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='lerobot/smolvla_base', cache_dir='${HF_HOME}')"

WORKDIR /workspace/project
CMD ["bash"]
