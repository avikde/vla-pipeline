# VLA Pipeline testing for Robot Arm

Vision-Language-Action model integration with low-level control using MuJoCo simulation.

Software: PyTorch 2.10.0 with CUDA 12.8, LeRobot 0.4.3 with X-VLA, MuJoCo.

Blog posts for context:
- https://www.avikde.me/p/the-architecture-behind-end-to-end
- https://www.avikde.me/p/debugging-as-architecture-insight
- https://open.substack.com/pub/minpower/p/a-coding-agent-equivalent-for-robotics?r=5vzx85&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true

## Install

```sh
git clone https://github.com/avikde/vla-pipeline.git
cd vla-pipeline
```

Linux/WSL system dependencies:
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-venv python3-pip build-essential git
```

Python setup:
```bash
python3.13 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install mujoco huggingface_hub hf_xet "lerobot[xvla]" google-genai
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
# Main demo (opens MuJoCo viewer)
python scripts/demo_widowx.py

# Flags:
# -p / --planner        Action source: xvla, hardcoded, gemini-er (default)
# -v / --verbose        Per-step debug output
# -d / --dry-run        Visualize trajectory without running IK/control
# -f / --free-cam       Free orbit camera (default: locked to primary camera)
# --step-size 0.005     Reference policy movement speed (m/step)
# --kp 10               Override proportional gain for arm actuators
```

For `-p gemini-er`, get an API key and set the `GEMINI_API_KEY` environment variable as described https://ai.google.dev/gemini-api/docs/api-key.

## Architecture

**Inference loop** (`demo_widowx.py`):
1. Render `primary` camera at 4:3 (342×256), squish to 256×256 to match BridgeData distortion; second image is black (BridgeData's cam2 is often black)
2. Pack images + 8D proprioceptive state (XYZ, RPY, pad, gripper) + language tokens into observation
3. Run X-VLA → 20D action vector (2 timesteps × 10D)
4. Decode each 10D slice: XYZ target + 6D rotation + gripper value
5. In dry-run mode, visualize the EE trajectory
6. In non-dry-run mode, close the control loop:
  a. Feed EE target to Jacobian IK (`widowx_control.py`) → joint angles → MuJoCo actuators
  b. To be implemented

**Key modules:**
- `scripts/xvla_policy.py` — policy loading, observation building, action decoding, MuJoCo utilities (EE pose, cube position). Also contains `generate_non_vla_reference()` fallback reach-and-grasp policy.
- `scripts/widowx_control.py` — `WidowXController` class: damped least-squares Jacobian IK, 6-DOF or position-only mode, gripper mapping, joint limit enforcement.
- `scripts/gemini_er_policy.py` — Gemini ER object detection + MuJoCo camera calibration (pixel→3D via ray-plane intersection).
- `assets/widowx/` — MuJoCo XML models; `widowx_vision_scene.xml` is the primary scene (wooden table, `primary` camera, `third_person` debug camera).

**Action representation (EE6D):** 10D per timestep = [x, y, z, r1x, r1y, r1z, r2x, r2y, r2z, gripper]. The 6D rotation uses two columns of the rotation matrix (third reconstructed via cross product).

**Model checkpoint:** `lerobot/xvla-widowx` (HuggingFace, fine-tuned on BridgeData). Downloaded automatically on first run.

## Jupyter Notebook

X-VLA with WidowX arm - prompt to trajectory visualization:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/avikde/vla-pipeline/blob/main/xvla_widowx_vis_traj.ipynb)

## Acknowledgements

- **LeRobot Framework:** Open-source robotics ML framework by HuggingFace
- **WidowX model:** From [google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/trossen_wx250s)
