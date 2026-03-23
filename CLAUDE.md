# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Integrates X-VLA (Vision-Language-Action) model inference with low-level robot control for the WidowX 250s arm in MuJoCo simulation. The goal is: given a natural language task and camera images, generate end-effector trajectories and execute them.

It ties to a series of blog posts, of which these two are helpful context:
- https://www.avikde.me/p/the-architecture-behind-end-to-end
- https://www.avikde.me/p/debugging-as-architecture-insight

## Setup

```bash
python3.13 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install mujoco huggingface_hub hf_xet "lerobot[xvla]"
# For NVIDIA GPU (run after lerobot):
pip uninstall torch torchvision -y && pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Verify:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import lerobot; print('LeRobot version:', lerobot.__version__)"
```

## Running Scripts

```bash
# Main VLA inference demo (opens MuJoCo viewer)
python scripts/demo_xvla_widowx_debug.py

# Flags:
# --no-vla              Use reference policy instead of X-VLA
# -v / --verbose        Per-step debug output
# -d / --dry-run        Visualize trajectory without running IK/control
# --step-size 0.005     Reference policy movement speed (m/step)
# --kp 10               Override proportional gain for arm actuators
```

No test suite or linter configured. Pyright is set up for type checking (standard mode).

## Architecture

**Inference loop** (`demo_xvla_widowx_debug.py`):
1. Render two 256×256 camera views from MuJoCo (`up` over-shoulder, `side`)
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
- `assets/widowx/` — MuJoCo XML models; `widowx_vision_scene.xml` is the primary scene used in demos (includes two cameras).

**Action representation (EE6D):** 10D per timestep = [x, y, z, r1x, r1y, r1z, r2x, r2y, r2z, gripper]. The 6D rotation uses two columns of the rotation matrix (third reconstructed via cross product).

**Model checkpoint:** `lerobot/xvla-widowx` (HuggingFace, fine-tuned on BridgeData). Downloaded automatically on first run.
