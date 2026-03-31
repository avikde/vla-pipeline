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
- "label": a short name (e.g. "red block", "dark cylinder", "blue target mat")
- "footprint": center location and size in [y, x, radius, height] format normalized to 0-1000
- "type": one of "block", "target", "obstacle"

Return JSON: [{"label": ..., "footprint": [y, x, radius, height], "type": ...}, ...]
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

    # Draw footprint: circle + vertical line up to height
    if "footprint" in det:
        fy, fx, fsize, fheight = det["footprint"]
        cx = fx / 1000 * w
        cy = fy / 1000 * h
        radius = fsize / 1000 * min(w, h)
        circle = mpatches.Circle(
            (cx, cy), radius,
            linewidth=2, edgecolor=color, facecolor=color, alpha=0.2,
        )
        ax.add_patch(circle)
        height_px = fheight / 1000 * h
        ax.plot([cx, cx], [cy + height_px * 0.5, cy - height_px * 0.5], "-", color=color, linewidth=2)

legend = [mpatches.Patch(color=c, label=t) for t, c in TYPE_COLORS.items()]
ax.legend(handles=legend, loc="upper right", fontsize=8)
ax.axis("off")
ax.set_title("Gemini ER detections")
plt.tight_layout()
plt.show()
