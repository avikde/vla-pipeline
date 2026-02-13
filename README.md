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


**Install LeRobot with SmolVLA support:**
```bash
pip install "lerobot[smolvla]"
```

**Install PyTorch 2.10.0 with CUDA 12.8 (for RTX 5070 Ti Blackwell support):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
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
