## Debugging scripts

### Setup

System dependencies:
```bash
# ffmpeg required by debug_bridgedata.py
sudo apt install -y ffmpeg        # Linux/WSL
brew install ffmpeg               # Mac
```

Python setup:
```bash
python3 -m venv venv
source venv/bin/activate
pip install mujoco huggingface_hub pandas pillow numpy google-genai matplotlib
```

### Running

Run from the repo root:
```bash
python debug/debug_bridgedata.py
python debug/gemini_probe.py [image.png]
```

- `debug_bridgedata.py`: Downloads a BridgeData episode from HuggingFace, renders the equivalent MuJoCo camera view at the home pose, and saves a side-by-side `camera_comparison.png` for visual alignment inspection. Accepts `--episode N` (default: 2076, "pick up the red cube").
- `gemini_probe.py`: Sends an image to Gemini ER and prints the raw JSON response, then plots the detected object footprints overlaid on the image. Defaults to `mujoco_primary_frame0.png` (which is saved by `debug_bridgedata.py`); pass a path as the first argument to use a different image. Requires `GEMINI_API_KEY` to be set.
