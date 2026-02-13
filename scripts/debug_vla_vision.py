#!/usr/bin/env python3
"""
Debug script to verify VLA vision pipelines.

Tests both raw and preprocessed (512x512 padded) images.

Some things to verify:
RGB ordering, resolution pipeline, and that the vision encoder
can perceive the scene objects

Supported models:
  --model smolvla     SmolVLM2-500M (used by SmolVLA and VLA-0-Smol)
  --model xvla        Qwen2.5-VL-3B (used by X-VLA, frozen during training)
  --model octo        No VLM backbone — saves image only
"""

import argparse
import mujoco
import numpy as np
import torch
from PIL import Image

parser = argparse.ArgumentParser(description='VLA Vision Debug')
parser.add_argument('--model', choices=['smolvla', 'xvla', 'octo'], default='smolvla',
                    help='Which VLA backbone to test (default: smolvla)')
args = parser.parse_args()

# VLM backbone for each VLA model
VLM_BACKBONES = {
    'smolvla': 'HuggingFaceTB/SmolVLM2-500M-Video-Instruct',
    'xvla': 'Qwen/Qwen2.5-VL-3B-Instruct',
    'octo': None,  # No pretrained VLM
}

# ── 1. Setup MuJoCo scene ────────────────────────────────────────────────────

print("Loading MuJoCo scene...")
model = mujoco.MjModel.from_xml_path('assets/so101/so101_vision_scene.xml')
data = mujoco.MjData(model)

model.vis.global_.offwidth = max(model.vis.global_.offwidth, 512)
model.vis.global_.offheight = max(model.vis.global_.offheight, 512)

renderer_main = mujoco.Renderer(model, height=512, width=512)
renderer_vla = mujoco.Renderer(model, height=256, width=256)

# Settle physics
for _ in range(100):
    mujoco.mj_step(model, data)

# Print cube position for reference
red_cube_id = model.body('red_cube').id
print(f"Red cube position: {data.xpos[red_cube_id]}")

# ── 2. Render camera images ──────────────────────────────────────────────────

def render_camera(camera_name):
    camera_id = model.camera(camera_name).id
    if camera_name == 'third_person':
        renderer_main.update_scene(data, camera=camera_id)
        pixels = renderer_main.render()
        pil_img = Image.fromarray(pixels)
        pil_img_resized = pil_img.resize((256, 256), Image.Resampling.BILINEAR)
        return np.array(pil_img_resized)
    else:
        renderer_vla.update_scene(data, camera=camera_id)
        return renderer_vla.render()

print("\nRendering cameras...")
img_third = render_camera('third_person')
img_top = render_camera('top_down')
print(f"  third_person: {img_third.shape}, top_down: {img_top.shape}")

Image.fromarray(img_third).save("debug_third_person.png")
print("  Saved debug_third_person.png")

# ── 3. Prepare preprocessed (padded) image ───────────────────────────────────

from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad

tensor_01 = torch.from_numpy(img_third).permute(2, 0, 1).float().unsqueeze(0) / 255.0
tensor_padded = resize_with_pad(tensor_01, 512, 512, pad_value=0)
padded_img_np = (tensor_padded[0].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)

print(f"  Preprocessed: {tensor_padded.shape}, range=[{tensor_padded.min():.3f}, {tensor_padded.max():.3f}]")

# ── 4. VLM visual grounding test ─────────────────────────────────────────────

vlm_name = VLM_BACKBONES[args.model]

if vlm_name is None:
    print(f"\n=== {args.model.upper()} has no pretrained VLM backbone ===")
    print("Perception debugging is limited to visual inspection of debug_third_person.png.")
    print("Octo uses a shallow CNN patch encoder trained from scratch — no standalone")
    print("vision model to query.")
else:
    print(f"\n=== VLM VISUAL GROUNDING TEST ({args.model}) ===")
    print(f"Loading {vlm_name}...")

    from transformers import AutoProcessor, AutoModelForImageTextToText

    vlm_processor = AutoProcessor.from_pretrained(vlm_name)
    vlm_model = AutoModelForImageTextToText.from_pretrained(
        vlm_name, dtype=torch.float32
    ).eval()

    def ask_vlm(pil_image, question):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        prompt = vlm_processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = vlm_processor(text=prompt, images=[pil_image], return_tensors="pt")
        with torch.inference_mode():
            output_ids = vlm_model.generate(**inputs, max_new_tokens=150)
        generated = vlm_processor.batch_decode(
            output_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )[0]
        return generated.strip()

    test_image = Image.fromarray(img_third)
    questions = [
        "Describe what you see in this image in detail.",
        "Is there a red cube in this image? If so, where is it?",
        "What objects do you see on the table?",
        "Do you see a robot arm? Describe its position.",
    ]

    print(f"\n--- Raw third_person image (256x256) ---")
    for question in questions:
        answer = ask_vlm(test_image, question)
        print(f"\n  Q: {question}")
        print(f"  A: {answer}")

    # Test on preprocessed (padded) image
    preprocessed_pil = Image.fromarray(padded_img_np)
    print(f"\n--- Preprocessed image (512x512 padded) ---")
    answer = ask_vlm(preprocessed_pil, "Is there a red cube in this image? Describe everything you see.")
    print(f"\n  Q: Is there a red cube in this image? Describe everything you see.")
    print(f"  A: {answer}")

    if args.model == 'xvla':
        print("\n  NOTE: X-VLA keeps this VLM frozen during training.")
        print("  These answers directly reflect the visual features X-VLA conditions on.")
    elif args.model == 'smolvla':
        print("\n  NOTE: SmolVLA fine-tunes the vision encoder end-to-end.")
        print("  Text QA quality does NOT directly predict action quality.")

print(f"\n{'='*60}")
print("Debug complete.")
print(f"{'='*60}")
