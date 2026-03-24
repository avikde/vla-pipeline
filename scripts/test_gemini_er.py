from google import genai
from google.genai import types
import sys
import json
import re
import matplotlib.pyplot as plt
from PIL import Image

# https://ai.google.dev/gemini-api/docs/robotics-overview

PROMPT = """
          Point to no more than 10 items in the image. The label returned
          should be an identifying name for the object detected.
          The answer should follow the json format: [{"point": <point>,
          "label": <label1>}, ...]. The points are in [y, x] format
          normalized to 0-1000.
        """
client = genai.Client()


# Load your image
with open(sys.argv[1], 'rb') as f:
    image_bytes = f.read()

image_response = client.models.generate_content(
    model="gemini-robotics-er-1.5-preview",
    contents=[
        types.Part.from_bytes(
            data=image_bytes,
            mime_type='image/png',
        ),
        PROMPT
    ],
    config = types.GenerateContentConfig(
        temperature=0.5,
        thinking_config=types.ThinkingConfig(thinking_budget=0)
    )
)

print(image_response.text)

# Parse JSON from response (strip markdown fences if present)
raw = image_response.text.strip()
json_match = re.search(r"```json?\s*(.*?)\s*```", raw, re.DOTALL)
detections = json.loads(json_match.group(1) if json_match else raw)

# Plot image with annotated points
img = Image.open(sys.argv[1])
w, h = img.size

fig, ax = plt.subplots()
ax.imshow(img)
for det in detections:
    y_norm, x_norm = det["point"]
    x_px = x_norm / 1000 * w
    y_px = y_norm / 1000 * h
    ax.plot(x_px, y_px, "ko", markersize=8)
    ax.annotate(det["label"], (x_px, y_px), textcoords="offset points",
                xytext=(8, -8), color="yellow", fontsize=9, fontweight="bold")
ax.axis("off")
plt.tight_layout()
plt.show()
