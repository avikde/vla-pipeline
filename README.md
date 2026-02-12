# VLA Pipeline for SO-101 Robot

Vision-Language-Action model integration with the SO-101 robot arm using MuJoCo simulation and SmolVLA.

Software used:
- JAX 0.9.0.1 with CUDA 12 support
- **PyTorch 2.10.0 with CUDA 12.8** (need CUDA 12.8 for Blackwell/sm_120 support)
- LeRobot 0.4.3 with SmolVLA
- MuJoCo + MJX

## Prerequisites

**Hardware:** NVIDIA GPU recommended (tested on RTX 5070ti mobile, 12GB VRAM)

**Software:**
- WSL2 (for Windows users) or Linux
- NVIDIA drivers installed
- Python 3.11+

**Verify GPU access in WSL2:**
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

### 2. Install System Dependencies

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-venv python3-pip build-essential git
```

### 3. Create Python Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install All Dependencies

**Install JAX with CUDA support:**
```bash
pip install --upgrade pip
pip install "jax[cuda12]"
```

**Install MuJoCo and MJX (GPU-accelerated physics):**
```bash
pip install mujoco mujoco-mjx
```

**Install PyTorch 2.10.0 with CUDA 12.8 (for RTX 5070 Ti Blackwell support):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

**Install LeRobot with SmolVLA support:**
```bash
pip install "lerobot[smolvla]"
```

**Install visualization and utilities:**
```bash
pip install dm_control matplotlib numpy pillow
```

### 5. Verify Installation

**Check JAX GPU access:**
```sh
python -c "import jax; print('JAX backend:', jax.default_backend()); print('JAX devices:', jax.devices())"
```
Expected output: Should show `cuda` backend and `CudaDevice`.

**Check PyTorch GPU access:**
```sh
python -c "import torch; print('PyTorch CUDA available:', torch.cuda.is_available())"
```
Expected output: `True`

**Check LeRobot installation:**
```sh
python -c "import lerobot; print('LeRobot version:', lerobot.__version__)"
```

## Quick Start

### Run SO-101 Simulation Demo

```sh
python scripts/demo_so101.py
```

This demonstrates the SO-101 robot with 3-camera vision setup and basic red cube detection.

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

## About SmolVLA

**SmolVLA** is a 450M parameter Vision-Language-Action model that:
- Is already pretrained on SO-101 robot data (no fine-tuning needed for basic tasks!)
- Trained on 487 community datasets including SO-101 manipulation tasks
- Uses ~2-3GB VRAM for inference (fits comfortably on 12GB GPU)
- Works out-of-the-box for pick-and-place, cube stacking, and other manipulation tasks

**Model:** `lerobot/smolvla_base` from Hugging Face

**Example tasks it can perform:**
- "Pick the red cube and place it on the blue cube"
- "Move the cube to the left"
- "Stack the cubes"

The model automatically downloads (~1.8GB) on first use.

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

