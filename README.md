# VLA Pipeline for SO-101 Robot

https://colab.research.google.com/github/avikde/vla-pipeline/blob/ipynb/xvla_widowx_vis_traj.ipynb


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

# For faster model downloads
pip install huggingface_hub[hf_xet]

# Install LeRobot with SmolVLA and X-VLA
pip install "lerobot[smolvla]"
pip install "lerobot[xvla]"
```

**CUDA support:** The LeRobot scripts install `torch 2.7.1` and `torchvision 0.22`, and with CPU support only. To utilize an NVIDIA GPU, we need to install torch with CUDA support. For my RTX 5070 Ti Blackwell GPU, I needed CUDA 12.8 for sm120 support. This should be run *after* the LeRobot packages.
```bash
pip uninstall torch torchvision -y && pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### Verify Installation


```sh
# Torch: should say "2.10.0+cu128" and  "True" for CUDA access
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# LeRobot
python -c "import lerobot; print('LeRobot version:', lerobot.__version__)"
```

<!--
Check JAX GPU access (**Skip for now**)
```sh
python -c "import jax; print('JAX backend:', jax.default_backend()); print('JAX devices:', jax.devices())"
```
-->

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

### Run X-VLA Inference Demo

```sh
python scripts/demo_xvla_widowx.py
```

This demonstrates X-VLA's modular soft prompt architecture using the WidowX robot with the `lerobot/xvla-widowx` checkpoint (fine-tuned on BridgeData). X-VLA only requires training 1% of parameters (9M) to adapt to new robot embodiments.

### Compare VLA Architectures

```sh
python scripts/compare_vla_modularity.py
```

This shows architectural differences in action space modularity between SmolVLA and X-VLA, helpful for understanding how to customize action spaces.

### Verify VLA Setup

```sh
python scripts/verify_vla_setup.py
```

This runs automated checks to ensure SmolVLA is properly installed and working.

## VLA Model Comparison

All models below fit on a 12GB GPU (RTX 5070 Ti) and are available through LeRobot.

| Model | Params | VRAM | Vision Encoder | Action Head | Vision Fine-tuned? | Params to Adapt |
|---|---|---|---|---|---|---|---|
| **X-VLA** | 900M | ~3-5GB | Pretrained VLM + shared ViT for aux views | Flow matching (10 steps) | Partially (VLM frozen, soft prompts trained) | **9M (1%)** |
| **SmolVLA** | 450M | ~2-3GB | SigLIP (via SmolVLM2-500M) | Flow matching (10 steps) | Yes (end-to-end) | **450M (100%)** |

**Key architectural differences:**

**Action Modularity** refers to how easily the model can adapt to different robot morphologies and action spaces:
- **X-VLA** uses a **soft prompt hub** with frozen VLM backbone. Only the embodiment-specific soft prompts (9M params, 1% of model) need training for new robots. Excellent for cross-embodiment transfer.
- **SmolVLA** uses **cross-attention** where action tokens attend to VLM features. More coupled architecture requiring end-to-end training (450M params) for new action spaces. Linear projection adapters provide some flexibility.

**Action Decoding:**
- Both models use **flow matching** (10 denoising steps) to refine noisy action vectors into clean continuous actions
- Flow matching is conditioned on vision+language embeddings from the VLM
- Enables high-frequency control (50 Hz) with smooth, continuous actions

**Vision Encoding:**
- **X-VLA**: VLM stays frozen during embodiment adaptation, preserving pretrained visual understanding
- **SmolVLA**: End-to-end fine-tuning of vision encoder is critical (freezing drops success 58% → 25%)

References: [SmolVLA](https://huggingface.co/papers/2506.01844) | [X-VLA](https://arxiv.org/html/2510.10274v1) | [X-VLA LeRobot Docs](https://huggingface.co/docs/lerobot/en/xvla)

### Model Selection

**SmolVLA** (`lerobot/smolvla_base`) is pretrained on 487 community datasets including SO-101 manipulation tasks and works out-of-the-box for pick-and-place tasks. Model auto-downloads (~1.8GB) on first use. Best choice for SO-101 robot. **It does not work without fine-tuning. Source: https://github.com/huggingface/lerobot/issues/1763, https://github.com/huggingface/lerobot/issues/2221#issuecomment-3413221473**

**X-VLA** offers superior modularity for customizing action spaces. With only 1% of parameters trainable (9M vs SmolVLA's 450M), it's ideal for adapting to new robot morphologies while preserving the pretrained vision encoder. Available checkpoints:
- `lerobot/xvla-widowx` - Fine-tuned on BridgeData for WidowX robot (use with [demo_xvla_widowx.py](scripts/demo_xvla_widowx.py))
- `lerobot/xvla_base` - Base model for fine-tuning on custom robots

## Acknowledgements

- **SO-101 Robot Models:** URDF and MuJoCo XML files sourced from [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)
- **SmolVLA Model:** Pre-trained model from [HuggingFace LeRobot](https://huggingface.co/lerobot/smolvla_base)
- **LeRobot Framework:** Open-source robotics ML framework by HuggingFace
