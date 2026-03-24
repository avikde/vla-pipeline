from google import genai
from google.genai import types
import sys
import json
import re
import matplotlib.pyplot as plt
from PIL import Image

# https://ai.google.dev/gemini-api/docs/robotics-overview

PROMPT = """
          Locate and point to the red block and the blue target. The label returned
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

# Extract detected coordinates for prompt2
img = Image.open(sys.argv[1])
w, h = img.size
block_det = next((d for d in detections if "red" in d["label"].lower()), detections[0])
target_det = next((d for d in detections if "blue" in d["label"].lower()), detections[-1])
block_y, block_x = block_det["point"]
target_y, target_x = target_det["point"]

prompt2 = f"""
    You are a robotic arm with six degrees-of-freedom. You have the
    following functions available to you:

    def move(x, y, high):
      # moves the arm to the given coordinates. The boolean value 'high' set
      to True means the robot arm should be lifted above the scene for
      avoiding obstacles during motion. 'high' set to False means the robot
      arm should have the gripper placed on the surface for interacting with
      objects.

    def setGripperState(opened):
      # Opens the gripper if opened set to true, otherwise closes the gripper

    Perform a pick and place operation where you pick up the red block at
    normalized coordinates ({block_x}, {block_y}) and place it on the blue
    target at normalized coordinates ({target_x}, {target_y}).
    Provide the sequence of function calls as a JSON list of objects, where
    each object has a "function" key (the function name) and an "args" key
    (a list of arguments for the function).
    Also, include your reasoning before the JSON output.
    For example:
    Reasoning: To pick up the block, I will first move the arm to a high
    position above the block, open the gripper, move down to the block,
    close the gripper, lift the arm, move to a high position above the bowl,
    move down to the bowl, open the gripper, and then lift the arm back to
    a high position.
"""

plan_response = client.models.generate_content(
    model="gemini-robotics-er-1.5-preview",
    contents=[
        types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
        prompt2,
    ],
    config=types.GenerateContentConfig(
        temperature=0.5,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    ),
)

print("\n--- Pick-and-place plan ---")
print(plan_response.text)

# Parse plan steps from response
plan_raw = plan_response.text.strip()
plan_json_match = re.search(r"```json?\s*(.*?)\s*```", plan_raw, re.DOTALL)
plan_steps = json.loads(plan_json_match.group(1) if plan_json_match else
                        re.search(r"\[.*\]", plan_raw, re.DOTALL).group(0))  # type: ignore[union-attr]

# Plot image with detections + plan steps
fig, ax = plt.subplots()
ax.imshow(img)

# Detection points
for det in detections:
    y_norm, x_norm = det["point"]
    x_px = x_norm / 1000 * w
    y_px = y_norm / 1000 * h
    ax.plot(x_px, y_px, "ko", markersize=8)
    ax.annotate(det["label"], (x_px, y_px), textcoords="offset points",
                xytext=(8, -8), color="yellow", fontsize=9, fontweight="bold")

# Plan step coordinates (only for move() calls which have x, y args)
for i, step in enumerate(plan_steps):
    func = step["function"]
    args = step["args"]
    last_arg = args[-1]
    label = f"{func}_{last_arg}"
    if func == "move" and len(args) >= 2:
        sx, sy = float(args[0]), float(args[1])
        x_px = sx / 1000 * w
        y_px = sy / 1000 * h
        ax.plot(x_px, y_px, "rs", markersize=6)
        ax.annotate(f"{i}: {label}", (x_px, y_px), textcoords="offset points",
                    xytext=(8, 8), color="cyan", fontsize=8, fontweight="bold")
    else:
        print(f"  Step {i}: {label} (no coordinates)")

ax.axis("off")
plt.tight_layout()
plt.show()
