#!/usr/bin/env python3
"""Compare MuJoCo camera views against BridgeData training frames.

Downloads a BridgeData episode with a pick-block task and renders a
side-by-side camera_comparison.png for visual inspection.

Usage:
    python scripts/debug_bridgedata.py
    python scripts/debug_bridgedata.py --episode 2076
"""

import argparse
import subprocess
import sys

import mujoco
import numpy as np
from huggingface_hub import hf_hub_download
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, "scripts")
import widowx_control as ctrl

REPO = "IPEC-COMMUNITY/bridge_orig_lerobot"

parser = argparse.ArgumentParser()
parser.add_argument("--episode", type=int, default=2076,
                    help="BridgeData episode index (default: 2076, 'pick up the red cube')")
args = parser.parse_args()

ep = args.episode
chunk = ep // 1000

# --- Download BridgeData episode ---
print(f"Downloading BridgeData episode {ep} (chunk {chunk})...")
paths = {}
for key, fname in [
    ("img0", f"videos/chunk-{chunk:03d}/observation.images.image_0/episode_{ep:06d}.mp4"),
    ("img1", f"videos/chunk-{chunk:03d}/observation.images.image_1/episode_{ep:06d}.mp4"),
    ("data", f"data/chunk-{chunk:03d}/episode_{ep:06d}.parquet"),
]:
    paths[key] = hf_hub_download(REPO, fname, repo_type="dataset")
    print(f"  {fname}")

# Extract first frame from each video
for cam, key in [("image_0", "img0"), ("image_1", "img1")]:
    out = f"bridgedata_{cam}_ep{ep}.png"
    subprocess.run(
        ["ffmpeg", "-y", "-i", paths[key], "-frames:v", "1", out],
        capture_output=True,
    )

# Read BridgeData state
df = pd.read_parquet(paths["data"])
bd_state = np.array(df["observation.state"].iloc[0])
print(f"\nBridgeData ep{ep} state:")
print(f"  x={bd_state[0]:.4f} y={bd_state[1]:.4f} z={bd_state[2]:.4f}")
print(f"  roll={bd_state[3]:.4f} pitch={bd_state[4]:.4f} yaw={bd_state[5]:.4f}")
print(f"  gripper={bd_state[7]:.4f}")

# --- Render MuJoCo cameras ---
print("\nRendering MuJoCo cameras...")
model = mujoco.MjModel.from_xml_path("web/widowx/widowx_vision_scene.xml")
data = mujoco.MjData(model)
data.qpos[:8] = model.keyframe("home").qpos[:8]
data.ctrl[:] = model.keyframe("home").ctrl
for _ in range(100):
    mujoco.mj_step(model, data)

# Render at 4:3 then squish to 256x256 to match BridgeData distortion
renderer = mujoco.Renderer(model, height=256, width=342)
renderer.update_scene(data, camera=model.camera("primary").id)
raw = renderer.render()
img = Image.fromarray(raw).resize((256, 256))
img.save("mujoco_primary_frame0.png")

mj_state = ctrl.get_ee_state_8d(model, data)
print(f"\nMuJoCo state:")
print(f"  x={mj_state[0]:.4f} y={mj_state[1]:.4f} z={mj_state[2]:.4f}")
print(f"  roll={mj_state[3]:.4f} pitch={mj_state[4]:.4f} yaw={mj_state[5]:.4f}")
print(f"  gripper={mj_state[7]:.4f}")

# --- Build comparison image (1x2: BridgeData vs MuJoCo) ---
W, H, GAP, HDR = 256, 256, 10, 25
bd0 = Image.open(f"bridgedata_image_0_ep{ep}.png").resize((W, H))
mj_primary = Image.open("mujoco_primary_frame0.png")

canvas = Image.new("RGB", (W * 2 + GAP, H + HDR), (255, 255, 255))
draw = ImageDraw.Draw(canvas)

draw.text((10, 5), f"BridgeData image_0 (ep{ep})", fill=(0, 0, 0))
canvas.paste(bd0, (0, HDR))
draw.text((W + GAP + 10, 5), 'MuJoCo "primary"', fill=(0, 0, 0))
canvas.paste(mj_primary, (W + GAP, HDR))

canvas.save("camera_comparison.png")
print("\nSaved camera_comparison.png")
