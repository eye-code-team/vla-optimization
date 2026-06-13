# =====================================================================
#  SmolVLA-LIBERO hierarchical action-aware pipeline — production image
#  Reproduces the working env: Ubuntu 22.04 + CUDA 12.4 + Python 3.10 +
#  torch 2.6.0+cu124 + lerobot 0.4.4 + robosuite/LIBERO/mujoco (EGL render).
#  Target GPU: NVIDIA (e.g. 2x RTX 4090). Needs nvidia-container-toolkit on host.
# =====================================================================
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    # Headless OpenGL for mujoco/robosuite off-screen rendering (LIBERO rollout).
    # EGL needs a GPU; if the host has no EGL, override to MUJOCO_GL=osmesa at run.
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    HF_HOME=/root/.cache/huggingface

# ── System libraries ──────────────────────────────────────────────────
#  python3.10 · git/ffmpeg · OpenGL/EGL/OSMesa (mujoco) · build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3.10-distutils \
        python3-pip git curl ca-certificates ffmpeg \
        libgl1 libglib2.0-0 libegl1 libgles2 libglew-dev libosmesa6 \
        libglfw3 libosmesa6-dev patchelf cmake build-essential pkg-config \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && python -m pip install --upgrade "pip<25" setuptools wheel \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/project

# ── 1) PyTorch matching the working env (CUDA 12.4) ───────────────────
RUN pip install torch==2.6.0 torchvision==0.21.0 \
        --index-url https://download.pytorch.org/whl/cu124

# ── 2) ML + sim stack (pinned) ────────────────────────────────────────
# lerobot installed --no-deps first: its declared constraints on huggingface-hub
# (<0.36.0) are outdated and conflict with the newer hub/transformers/datasets
# versions that the scripts actually need. The lerobot 0.4.4 code itself works
# fine with newer hub; only the metadata constraint is wrong.
RUN pip install --no-deps lerobot==0.4.4
# rerun-sdk requires numpy>=2 which conflicts with the rest of the stack.
# Install --no-deps: only the rerun visualisation code is needed, not its numpy.
RUN pip install --no-deps "rerun-sdk>=0.24.0,<0.27.0"
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# ── 3) LIBERO benchmark — setup.py is broken (doesn't declare packages),
#       so we copy the Python package directly into dist-packages.
RUN git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git /tmp/LIBERO \
    && cp -r /tmp/LIBERO/libero /usr/local/lib/python3.10/dist-packages/libero \
    && pip install --force-reinstall robosuite==1.4.1 bddl==3.6.0 \
    && rm -rf /tmp/LIBERO \
    # PyTorch 2.6 changed weights_only default to True; patch LIBERO's torch.load calls
    && sed -i 's/torch\.load(init_states_path)/torch.load(init_states_path, weights_only=False)/g' \
       /usr/local/lib/python3.10/dist-packages/libero/libero/benchmark/__init__.py \
    && grep -rn "torch\.load(" /usr/local/lib/python3.10/dist-packages/libero/libero/ \
       | grep -v "weights_only" \
       | awk -F: '{print $1}' | sort -u \
       | xargs -I{} sed -i 's/torch\.load(\(.*\))/torch.load(\1, weights_only=False)/g' {} \
       || true

# ── 4) Project entrypoint + env check ─────────────────────────────────
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
COPY check_env.py  /opt/check_env.py
RUN chmod +x /usr/local/bin/entrypoint.sh

# Project code is bind-mounted at /workspace/project by docker-compose.
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
