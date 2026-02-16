#!/usr/bin/env python3
"""Render the two VLA camera views and save as a side-by-side image."""

import mujoco
import numpy as np
from PIL import Image

xml_path = 'assets/widowx/widowx_vision_scene.xml'
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Step simulation a bit so the scene settles
mujoco.mj_step(model, data)

W, H = 256, 256
model.vis.global_.offwidth = max(model.vis.global_.offwidth, W)
model.vis.global_.offheight = max(model.vis.global_.offheight, H)
renderer = mujoco.Renderer(model, height=H, width=W)

def render_camera(name):
    cam_id = model.camera(name).id
    renderer.update_scene(data, camera=cam_id)
    return renderer.render()

img_up = render_camera('up')
img_side = render_camera('side')

# Side-by-side with labels
combined = Image.new('RGB', (W * 2 + 20, H + 30), color=(255, 255, 255))
combined.paste(Image.fromarray(img_up), (0, 30))
combined.paste(Image.fromarray(img_side), (W + 20, 30))

# Add labels using simple text (no font dependency)
from PIL import ImageDraw
draw = ImageDraw.Draw(combined)
draw.text((W // 2 - 30, 5), 'image (over the shoulder)', fill=(0, 0, 0))
draw.text((W + 20 + W // 2 - 30, 5), 'image2 (side)', fill=(0, 0, 0))

out_path = 'camera_views.png'
combined.save(out_path)
print(f'Saved to {out_path}')
