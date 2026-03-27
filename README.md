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

### Python demo (opens MuJoCo viewer)

```bash
python scripts/demo_widowx.py

# Flags:
# -p / --planner        Action source: xvla, hardcoded, gemini-er (default)
# -v / --verbose        Per-step debug output
# -d / --dry-run        Visualize trajectory without running IK/control
# -f / --free-cam       Free orbit camera (default: locked to primary camera)
# --step-size 0.005     Reference policy movement speed (m/step)
# --kp 10               Override proportional gain for arm actuators
```

For `-p gemini-er`, get an API key and set the `GEMINI_API_KEY` environment variable as described in https://ai.google.dev/gemini-api/docs/api-key.

### Browser demo (GitHub Pages)

The `docs/` directory contains a fully client-side port of the Gemini ER pick-and-place demo using MuJoCo WASM + Three.js. No backend required.

```bash
cd docs && python3 serve.py
# Open http://localhost:8080
```

MuJoCo WASM requires `SharedArrayBuffer`, so the server must send COOP/COEP headers (`serve.py` handles this; `python3 -m http.server` won't work).

Visitors enter their own [Gemini API key](https://ai.google.dev/gemini-api/docs/api-key) (free tier).

## Architecture

### Inference loop (`demo_widowx.py`)
1. Render `primary` camera at 4:3 (342x256), squish to 256x256 to match BridgeData distortion
2. Pack images + 8D proprioceptive state (XYZ, RPY, pad, gripper) + language tokens into observation
3. Run X-VLA or Gemini ER planner -> 20D action vector (2 timesteps x 10D)
4. Feed EE target to Jacobian IK (`widowx_control.py`) -> joint angles -> MuJoCo actuators

### Key modules

| Module | Role |
|--------|------|
| `scripts/xvla_policy.py` | Policy loading, observation building, action decoding |
| `scripts/widowx_control.py` | `WidowXController`: damped least-squares Jacobian IK, 6-DOF, gripper mapping |
| `scripts/gemini_er_policy.py` | Gemini ER object detection + camera calibration (pixel->3D ray-plane intersection) |
| `docs/widowx/` | MuJoCo XML models + STL meshes; `widowx_vision_scene.xml` is the primary scene |

### Browser demo (`docs/`)

| Module | Role |
|--------|------|
| `docs/main.js` | Entry point: init, Gemini pipeline, waypoint sequencing, animation loop |
| `docs/mujoco-scene.js` | MuJoCo WASM init, Three.js rendering, MjvScene sync |
| `docs/ik-solver.js` | JS port of `WidowXController` |
| `docs/gemini-er.js` | JS port of `gemini_er_policy.py` + pre-baked fallback plan |
| `docs/math-utils.js` | Linear algebra, rotation math, pixel-to-3D projection |

Stack: [`@mujoco/mujoco`](https://www.npmjs.com/package/@mujoco/mujoco) WASM (CDN), [Three.js](https://threejs.org/) v0.170 (CDN), Gemini API via `fetch()`.

### Shared

**Action representation (EE6D):** 10D per timestep = [x, y, z, r1x, r1y, r1z, r2x, r2y, r2z, gripper]. The 6D rotation uses two columns of the rotation matrix (third reconstructed via cross product).

**Model checkpoint:** `lerobot/xvla-widowx` (HuggingFace, fine-tuned on BridgeData). Downloaded automatically on first run.

## Jupyter Notebook

X-VLA with WidowX arm - prompt to trajectory visualization:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/avikde/vla-pipeline/blob/main/xvla_widowx_vis_traj.ipynb)

## Acknowledgements

- **LeRobot Framework:** Open-source robotics ML framework by HuggingFace
- **WidowX model:** From [google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/trossen_wx250s)
