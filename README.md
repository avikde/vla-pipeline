# VLA Pipeline for SO-101 Robot

Vision-Language-Action model integration with the SO-101 robot arm using MuJoCo simulation and SmolVLA.

Software used:
- **PyTorch 2.10.0 with CUDA 12.8** (need CUDA 12.8 for Blackwell/sm_120 support)
- LeRobot 0.4.3 with SmolVLA
- JAX 0.9.0.1 with CUDA 12 support + MJX (**TODO**)
- MuJoCo

## Prerequisites

**Hardware:** NVIDIA GPU recommended (tested on RTX 5070ti mobile, 12GB VRAM)

**Software:**
- Windows, WSL2 or Linux
- NVIDIA drivers installed
- Python 3.11+

```sh
nvidia-smi
```
You should see your NVIDIA GPU listed. If not, update your Windows NVIDIA drivers.

## Installation

### 1. Clone Repository

```sh
git clone https://github.com/avikde/vla-pipeline.git
cd vla-pipeline
```

If using **Linux / WSL**, install these system dependencies. Skip if **Windows**:

```bash
# 2
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-venv python3-pip build-essential git
```

### 

```bash
# Create Python Virtual Environment
python3 -m venv venv
source venv/bin/activate

# Install JAX with CUDA support:
pip install --upgrade pip
pip install "jax[cuda12]"

# Install MuJoCo and MJX (GPU-accelerated physics)
pip install mujoco # mujoco-mjx

# Install visualization and utilities
pip install dm_control matplotlib numpy pillow

# Install LeRobot with SmolVLA support
pip install "lerobot[smolvla]"
```

For an NVIDIA GPU, install torch with CUDA support. For my RTX 5070 Ti Blackwell GPU, I needed CUDA 12.8 for sm120 support:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### Verify Installation


```sh
# Torch: should say "2.10.0+cu128 True" for CUDA access
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# LeRobot
python -c "import lerobot; print('LeRobot version:', lerobot.__version__)"
```

Check JAX GPU access (**Skip for now**)
```sh
python -c "import jax; print('JAX backend:', jax.default_backend()); print('JAX devices:', jax.devices())"
```

## Quick Start

### Run Mujoco benchmark without VLA

This demonstrates the SO-101 robot with 3-camera vision setup:
```sh
python scripts/bench_mujoco.py
```

### Run SmolVLA Inference Demo

```sh
python scripts/demo_vla_inference.py
```

This demonstrates language-conditioned robot control using SmolVLA (e.g., "Pick the red cube").

### Verify VLA Setup

```sh
python scripts/verify_vla_setup.py
```

This runs automated checks to ensure SmolVLA is properly installed and working.

## VLA Model Comparison

All models below fit on a 12GB GPU (RTX 5070 Ti).

| Model | Params | VRAM | Vision Encoder | Action Head | Vision Fine-tuned? |
|---|---|---|---|---|---|
| **SmolVLA** | 450M | ~2-3GB | SigLIP (via SmolVLM2-500M) | Flow matching (10 steps) | Yes (end-to-end) |
| Octo-Base | 93M | ~1-2GB | Shallow CNN patch encoder (no pretrained ViT) | Diffusion (20 steps) | N/A (trained from scratch) |
| X-VLA-0.9B | 900M | ~3-5GB | Pretrained VLM + shared ViT for aux views | Flow matching (10 steps) | Partially (VLM frozen, soft prompts trained) |
| VLA-0-Smol | 500M | ~2-3GB | SigLIP (via SmolVLM2-500M) | Autoregressive tokens (actions discretized into bins) | Yes (critical — freezing drops success 58% to 25%) |

**Key architectural differences:**
- **SmolVLA** and **X-VLA** use **flow matching**: a denoising loop refines noisy action vectors into clean continuous actions, conditioned on vision+language embeddings. The vision encoder provides spatial features, not text.
- **Octo** uses **diffusion** (similar concept to flow matching): a 3-layer MLP denoises actions conditioned on transformer embeddings. No pretrained VLM — all parameters trained from scratch on robot data.
- **VLA-0-Smol** uses **autoregressive token prediction**: actions are discretized into bins and generated as text tokens ("227 232 223 191"). This is the only architecture where text generation quality directly reflects action quality.

References: [SmolVLA](https://huggingface.co/papers/2506.01844) | [Octo](https://octo-models.github.io/) | [X-VLA](https://thu-air-dream.github.io/X-VLA/) | [VLA-0-Smol](https://robot-learning-collective.github.io/vla-0-smol)

### Current model: SmolVLA

**SmolVLA** is pretrained on 487 community datasets including SO-101 manipulation tasks and works out-of-the-box for pick-and-place tasks. Model `lerobot/smolvla_base` auto-downloads (~1.8GB) on first use.

## SO-101 Robot Specifications

- **Type:** 6-DOF robotic arm with gripper
- **Joints:** 5 servo joints + 1 gripper
- **Action Space:** 6D (auto-padded to 20D for VLA compatibility)
- **Cameras:** 3 views (third_person, top_down, wrist_cam) at 640x480
- **Base Model:** LeRobot SO-100 design

## GPU Considerations

**Performance:**
- SmolVLA inference: ~2-3GB VRAM
- MJX physics: Batch size 256 fits comfortably in 12GB VRAM
- Combined GPU usage: ~5-8GB during active VLA + physics

**Performance tips:**
- Plug in laptop to get max GPU power (important for mobile GPUs)
- Monitor GPU usage with `nvidia-smi` during training

Performance Summary on my laptop (Zen 5 CPU, RTX 5070ti mobile GPU):
- GPU: ~6-7ms average (from earlier test)
- CPU: ~3-4ms average (post-warmup)
- CPU cold-start: ~2.8 seconds (one-time penalty)
- GPU cold-start: ~7-10ms (minimal penalty)

## Advanced: Fine-tuning (Optional)

SmolVLA is already trained on SO-101 data, so fine-tuning is **only needed** if you want to teach it custom tasks not in the training data.

**Requirements for fine-tuning:**
- Physical SO-101 robot with teleoperation setup
- 50+ episodes of demonstration data
- Dataset in RLDS format via LeRobot

**See LeRobot documentation for details:**
- [LeRobot Docs](https://huggingface.co/docs/lerobot)
- [SmolVLA Paper](https://huggingface.co/papers/2506.01844)
- [SO-101 in LeRobot](https://huggingface.co/docs/lerobot/so101)

## Acknowledgements

- **SO-101 Robot Models:** URDF and MuJoCo XML files sourced from [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)
- **SmolVLA Model:** Pre-trained model from [HuggingFace LeRobot](https://huggingface.co/lerobot/smolvla_base)
- **LeRobot Framework:** Open-source robotics ML framework by HuggingFace


WSL

  Rendering (3 cameras)......... 1496.73 ms  ( 99.9%)
    getCameraImage()............ 1414.27 ms  ( 94.4%)
  Physics step..................    0.70 ms  (  0.0%)
  Total iteration............... 1498.36 ms  (100.0%)
    (min/max)...................  359.14 / 1674.81 ms

WSL opengl

  Rendering (3 cameras).........  884.64 ms  ( 99.8%)
    getCameraImage()............  800.09 ms  ( 90.3%)
  Physics step..................    0.73 ms  (  0.1%)
  Total iteration...............  886.39 ms  (100.0%)
    (min/max)...................  820.23 / 967.92 ms

WSL opengl single camera

  Rendering (1 camera)..........  335.89 ms  ( 99.5%)
    getCameraImage()............  279.10 ms  ( 82.6%)
  Physics step..................    0.76 ms  (  0.2%)
  Total iteration...............  337.72 ms  (100.0%)
    (min/max)...................  294.59 / 477.17 ms

Win

  Rendering (3 cameras).........  913.59 ms  ( 97.2%)
    getCameraImage()............  833.40 ms  ( 88.7%)
  Physics step..................    0.62 ms  (  0.1%)
  Total iteration...............  939.45 ms  (100.0%)
    (min/max)...................  898.33 / 1425.80 ms

Win opengl

  Rendering (3 cameras).........  159.11 ms  ( 98.5%)
    getCameraImage()............   76.99 ms  ( 47.6%)
  Physics step..................    0.60 ms  (  0.4%)
  Total iteration...............  161.59 ms  (100.0%)
    (min/max)...................  142.42 / 257.57 ms

Win opengl single camera

  Rendering (1 camera)..........   99.17 ms  ( 97.7%)
    getCameraImage()............   41.72 ms  ( 41.1%)
  Physics step..................    0.73 ms  (  0.7%)
  Total iteration...............  101.50 ms  (100.0%)
    (min/max)...................   80.99 / 156.19 ms

Mujoco 

  Rendering (3 cameras).........  859.79 ms  ( 99.1%)
    render() calls..............  855.82 ms  ( 98.7%)
  Viewer sync...................    3.16 ms  (  0.4%)
  Physics step..................    0.50 ms  (  0.1%)
  Total iteration...............  867.33 ms  (100.0%)
    (min/max)...................  786.47 / 1428.28 ms

Mujoco single camera

  Rendering (1 camera)..........  328.32 ms  ( 98.4%)
    render() calls..............  325.35 ms  ( 97.5%)
  Viewer sync...................    4.34 ms  (  1.3%)
  Physics step..................    0.29 ms  (  0.1%)
  Total iteration...............  333.83 ms  (100.0%)
    (min/max)...................  303.48 / 631.69 ms

Win Mujoco 

  Rendering (3 cameras).........   47.22 ms  ( 92.7%)
    render() calls..............   43.41 ms  ( 85.2%)
  Viewer sync...................    3.16 ms  (  6.2%)
  Physics step..................    0.22 ms  (  0.4%)
  Total iteration...............   50.95 ms  (100.0%)
    (min/max)...................   48.31 / 143.82 ms

Win Mujoco single camera

  Rendering (1 camera)..........   15.09 ms  ( 85.3%)
    render() calls..............   11.45 ms  ( 64.7%)
  Viewer sync...................    1.86 ms  ( 10.5%)
  Physics step..................    0.39 ms  (  2.2%)
  Total iteration...............   17.70 ms  (100.0%)
    (min/max)...................   13.85 / 122.73 ms


demo_vla_inference (CPU)

  Rendering (3 cameras).........   40.72 ms  ( 37.2%)
    render() calls..............   37.44 ms  ( 34.2%)
  Image preprocessing...........    2.15 ms  (  2.0%)
  VLA inference.................   64.47 ms  ( 58.9%)
  Physics step..................    0.18 ms  (  0.2%)
  Viewer sync...................    1.77 ms  (  1.6%)
  Total iteration...............  109.42 ms  (100.0%)
    (min/max)...................   46.00 / 3195.77 ms

demo_vla_inference (GPU)

  Rendering (3 cameras).........   74.91 ms  ( 62.9%)
    render() calls..............   71.18 ms  ( 59.8%)
  Image preprocessing...........    4.87 ms  (  4.1%)
  VLA inference.................   36.55 ms  ( 30.7%)
  Physics step..................    0.26 ms  (  0.2%)
  Viewer sync...................    2.00 ms  (  1.7%)
  Total iteration...............  119.01 ms  (100.0%)
    (min/max)...................   50.94 / 2594.17 ms
