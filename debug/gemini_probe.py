"""Send a single image to Gemini ER and print the response. Usage:
  python scripts/gemini_probe.py [image.png]
Defaults to mujoco_primary_frame0.png in the repo root.
"""

import io
import json
import re
import sys
import time

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from google import genai
from google.genai import types
from PIL import Image

PROMPT = """
Detect all objects and obstacles on the table. For each, return:
- "label": a short identifying name (e.g. "red block", "dark cylinder", "blue target mat")
- "point": center location in [y, x] format normalized to 0-1000
- "box_2d": bounding box as [top_y, left_x, bottom_y, right_x] normalized to 0-1000
- "type": one of "block", "target", "obstacle"

Do not include the robot gripper.

Return JSON: [{"label": ..., "point": [y, x], "box_2d": [top_y, left_x, bottom_y, right_x], "type": ...}, ...]
"""

TYPE_COLORS = {"block": "red", "target": "blue", "obstacle": "black"}

image_path = sys.argv[1] if len(sys.argv) > 1 else "mujoco_primary_frame0.png"

img = Image.open(image_path).convert("RGB")
buf = io.BytesIO()
img.save(buf, format="JPEG", quality=90)
image_bytes = buf.getvalue()
print(f"Image: {image_path}  ({img.width}x{img.height})")

client = genai.Client()
print("Sending to Gemini ER...")
t0 = time.monotonic()
response = client.models.generate_content(
    model="gemini-robotics-er-1.5-preview",
    contents=[
        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
        PROMPT,
    ],
    config=types.GenerateContentConfig(
        temperature=0.5,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    ),
)
print(f"Gemini responded in {time.monotonic() - t0:.1f}s")
print(response.text)

# Parse JSON from response
raw = response.text.strip()
json_match = re.search(r"```json?\s*(.*?)\s*```", raw, re.DOTALL)
detections = json.loads(json_match.group(1) if json_match else raw)

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(img)
w, h = img.width, img.height
for det in detections:
    color = TYPE_COLORS.get(det.get("type", ""), "white")

    # Draw bounding box
    if "box_2d" in det:
        top_y, left_x, bot_y, right_x = det["box_2d"]
        x0, y0 = left_x / 1000 * w, top_y / 1000 * h
        bw, bh = (right_x - left_x) / 1000 * w, (bot_y - top_y) / 1000 * h
        rect = mpatches.Rectangle(
            (x0, y0), bw, bh,
            linewidth=2, edgecolor=color, facecolor=color, alpha=0.15,
        )
        ax.add_patch(rect)

    # Draw center point and label
    if "point" in det:
        py, px = det["point"]
        cx, cy = px / 1000 * w, py / 1000 * h
        ax.plot(cx, cy, "o", color=color, markersize=6)
        ax.text(cx + 4, cy - 4, det.get("label", ""), color=color, fontsize=7,
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "alpha": 0.6, "ec": "none"})

legend = [mpatches.Patch(color=c, label=t) for t, c in TYPE_COLORS.items()]
ax.legend(handles=legend, loc="upper right", fontsize=8)
ax.axis("off")
ax.set_title("Gemini ER detections")
plt.tight_layout()
plt.show()
