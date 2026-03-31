# VLA Pipeline testing for Robot Arm

Vision-Language-Action model integration with low-level control using MuJoCo simulation.

Software: PyTorch 2.10.0 with CUDA 12.8, LeRobot 0.4.3 with X-VLA, MuJoCo.

Blog posts for context:
- [The architecture behind “end-to-end” robotics pipelines](https://www.avikde.me/p/the-architecture-behind-end-to-end)
- [Debugging as architecture insight: dissecting a VLA
](https://www.avikde.me/p/debugging-as-architecture-insight)
- [A coding agent equivalent for robotics pipelines
](https://www.avikde.me/p/a-coding-agent-equivalent-for-roboticse)

## Web demo (start here)

Try the browser-based demo with MuJoCo WASM + Three.js, no installation required:
- Grab your own [Gemini API key](https://ai.google.dev/gemini-api/docs/api-key) (free tier), or use the pre-baked fallback plan
- Open https://avikde.github.io/vla-pipeline/ in Chrome
- Click "Run Demo" or "Use Cached Plan" and watch the pick-and-place in action!
- Use the mouse to orbit the camera, and check the console for debug logs

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

The `docs/` directory contains a fully client-side embodied reasoning demo using MuJoCo WASM + Three.js. No backend required.

```bash
node docs/serve.js
# Open http://localhost:8080
```

## Architecture

### Browser demo (`docs/`)

| Module | Role |
|--------|------|
| `docs/main.js` | Entry point: init, Gemini pipeline, waypoint sequencing, animation loop |
| `docs/mujoco-scene.js` | MuJoCo WASM init, Three.js rendering, MjvScene sync |
| `docs/ik-solver.js` | JS port of `WidowXController` |
| `docs/gemini-er.js` | JS port of `gemini_er_policy.py` + pre-baked fallback plan |
| `docs/math-utils.js` | Linear algebra, rotation math, pixel-to-3D projection |

Stack: [`@mujoco/mujoco`](https://www.npmjs.com/package/@mujoco/mujoco) WASM (CDN), [Three.js](https://threejs.org/) v0.170 (CDN), Gemini API via `fetch()`.

### WidowX

**Action representation (EE6D):** 10D per timestep = [x, y, z, r1x, r1y, r1z, r2x, r2y, r2z, gripper]. The 6D rotation uses two columns of the rotation matrix (third reconstructed via cross product).

## Acknowledgements

- **LeRobot Framework:** Open-source robotics ML framework by HuggingFace
- **WidowX model:** From [google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/trossen_wx250s)
