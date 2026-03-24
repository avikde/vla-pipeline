# VLA Pipeline testing for Robot Arm

Vision-Language-Action model integration with low-level control using MuJoCo simulation.

Software: PyTorch 2.10.0 with CUDA 12.8, LeRobot 0.4.3 with X-VLA, MuJoCo.

## Jupyter Notebook (start here)

X-VLA with WidowX arm - prompt to trajectory visualization:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/avikde/vla-pipeline/blob/main/xvla_widowx_vis_traj.ipynb)

## Local Installation

```sh
git clone https://github.com/avikde/vla-pipeline.git
cd vla-pipeline
```

Linux/WSL system dependencies:
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-venv python3-pip build-essential git
```

Mac: `brew install bazel`

Python setup:
```bash
python3.13 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install mujoco huggingface_hub hf_xet "lerobot[xvla]"
# For NVIDIA GPU (run after lerobot):
pip uninstall torch torchvision -y && pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

On Windows, use `python` not `python3`. On Mac, use `mjpython`.

Verify:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import lerobot; print('LeRobot version:', lerobot.__version__)"
```

## Running

```bash
# Main VLA inference demo (opens MuJoCo viewer)
python scripts/demo_xvla_widowx_debug.py

# Flags:
# --no-vla              Use reference policy instead of X-VLA
# -v / --verbose        Per-step debug output
# -d / --dry-run        Visualize trajectory without running IK/control
# -f / --free-cam       Free orbit camera (default: locked to primary camera)
# --step-size 0.005     Reference policy movement speed (m/step)
# --kp 10               Override proportional gain for arm actuators
```

## Gemini ER

```shell
pip install google-genai
```

Get an API key and set the `GEMINI_API_KEY` environment variable as described https://ai.google.dev/gemini-api/docs/api-key.

## Acknowledgements

- **LeRobot Framework:** Open-source robotics ML framework by HuggingFace
- **WidowX model:** From [google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/trossen_wx250s)
