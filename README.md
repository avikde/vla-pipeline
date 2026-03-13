# VLA Pipeline testing for Robot Arm

Vision-Language-Action model integration with low-level control using MuJoCo simulation.

Software used:
- **PyTorch 2.10.0 with CUDA 12.8** (need CUDA 12.8 for Blackwell/sm_120 support)
- LeRobot 0.4.3 with X-VLA
- MuJoCo

<!-- - JAX 0.9.0.1 with CUDA 12 support + MJX (**TODO**) -->

## Jupyter Notebook (start here)

👇 X-VLA with WidowX arm - prompt to trajectory visualization

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/avikde/vla-pipeline/blob/main/xvla_widowx_vis_traj.ipynb)

## Local Installation

### Clone Repository

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

Mac
```sh
brew install bazel
```

### Python dependencies

```bash
# Create Python Virtual Environment
python3.13 -m venv venv
source venv/bin/activate

# Install JAX with CUDA support:
pip install --upgrade pip
# pip install "jax[cuda12]"

# Install MuJoCo and MJX (GPU-accelerated physics)
pip install mujoco # mujoco-mjx

# For faster model downloads
pip install huggingface_hub hf_xet

# Install LeRobot with VLAs
# pip install "lerobot[smolvla]"
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

## Scripts to run

### Run X-VLA Inference Demo

```sh
python scripts/demo_xvla_widowx.py
```

This demonstrates X-VLA's modular soft prompt architecture using the WidowX robot with the `lerobot/xvla-widowx` checkpoint (fine-tuned on BridgeData).

## Acknowledgements

<!-- - **SO-101 Robot Models:** URDF and MuJoCo XML files sourced from [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)
- **SmolVLA Model:** Pre-trained model from [HuggingFace LeRobot](https://huggingface.co/lerobot/smolvla_base) -->
- **LeRobot Framework:** Open-source robotics ML framework by HuggingFace
- **WidowX model:** From [google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/trossen_wx250s)
