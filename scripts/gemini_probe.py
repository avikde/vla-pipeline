"""Send a single image to Gemini ER and print the response. Usage:
  python scripts/gemini_probe.py [image.png]
Defaults to mujoco_primary_frame0.png in the repo root.
"""

import io
import sys
import time

from google import genai
from google.genai import types
from PIL import Image

PROMPT = """Describe every object and obstacle on the table. For each, return:
- "label": a short name (e.g. "red block", "dark cylinder", "blue target mat")
- "point": center location in [y, x] format normalized to 0-1000
- "type": one of "block", "target", "obstacle"

Return as JSON: [{"label": ..., "point": [y, x], "type": ...}, ...]"""

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
