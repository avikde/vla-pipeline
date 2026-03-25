"""Gemini ER object detection + MuJoCo camera calibration for pixel-to-3D projection."""

import io
import json
import re

import mujoco
import numpy as np
from google import genai
from google.genai import types
from PIL import Image


def pixel_to_world_3d(
    pixel_xy: tuple[float, float],
    model,
    data,
    camera_name: str = "primary",
    render_size: tuple[int, int] = (342, 256),
    vla_size: tuple[int, int] = (256, 256),
    table_z: float = 0.02,
) -> np.ndarray:
    """Project a pixel (in VLA image space) to 3D world coords via ray-plane intersection.

    Args:
        pixel_xy: (x, y) in the VLA-sized (squished) image.
        model: MjModel (for camera fovy).
        data: MjData (for camera pose after mj_forward).
        camera_name: MuJoCo camera name.
        render_size: (width, height) of the pre-squish render.
        vla_size: (width, height) of the squished VLA image.
        table_z: height of the table plane for ray intersection.
    """
    cam_id = model.camera(camera_name).id
    render_w, render_h = render_size
    vla_w, vla_h = vla_size

    # Camera intrinsics from fovy
    fovy_rad = np.deg2rad(model.cam_fovy[cam_id])
    fy = (render_h / 2) / np.tan(fovy_rad / 2)
    fx = fy  # MuJoCo uses square pixels
    cx_render = render_w / 2
    cy_render = render_h / 2

    # Convert pixel from VLA (squished) space back to render space
    u_vla, v_vla = pixel_xy
    u_render = u_vla * (render_w / vla_w)
    v_render = v_vla * (render_h / vla_h)

    # Camera extrinsics from MuJoCo (after mj_forward)
    cam_pos = data.cam_xpos[cam_id].copy()
    cam_rot = data.cam_xmat[cam_id].reshape(3, 3).copy()  # columns = camera axes in world

    # MuJoCo/OpenGL convention: camera looks along -Z, Y is up, X is right.
    # Pixel (u, v) with origin top-left maps to camera-frame direction:
    #   d_cam = ( (u - cx) / fx,  -(v - cy) / fy,  -1 )
    # The v-axis is negated because pixel row increases downward but camera Y is up.
    d_cam = np.array([
        (u_render - cx_render) / fx,
        -(v_render - cy_render) / fy,
        -1.0,
    ])

    # Transform ray direction to world frame
    # cam_rot columns are the camera's X, Y, Z axes in world frame
    d_world = cam_rot @ d_cam
    d_world = d_world / np.linalg.norm(d_world)

    # Ray-plane intersection: cam_pos + t * d_world, solve for z = table_z
    if abs(d_world[2]) < 1e-8:
        print("  ⚠️  Ray is parallel to table plane, cannot intersect")
        return cam_pos  # fallback
    t = (table_z - cam_pos[2]) / d_world[2]
    world_point = cam_pos + t * d_world

    return world_point


HEIGHT_OFFSET = 0.15  # metres above table for "high" moves
GRASP_HEIGHT = 0.04   # z for "low" moves (top of block)
TABLE_Z = 0.02        # table surface z


def _encode_image(image_rgb: np.ndarray) -> bytes:
    """Encode numpy RGB array to JPEG bytes."""
    img_pil = Image.fromarray(image_rgb)
    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG")
    return buf.getvalue()


def _gemini_call(image_bytes: bytes, prompt: str) -> str:
    """Send image + prompt to Gemini ER and return raw text response."""
    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-robotics-er-1.5-preview",
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            prompt,
        ],
        config=types.GenerateContentConfig(
            temperature=0.5,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return response.text


def _parse_json(raw_text: str):
    """Parse JSON from Gemini response, stripping markdown fences if present."""
    raw = raw_text.strip()
    json_match = re.search(r"```json?\s*(.*?)\s*```", raw, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(1))
    # Fallback: find first JSON array or object
    arr_match = re.search(r"[\[{].*[\]}]", raw, re.DOTALL)
    if arr_match:
        return json.loads(arr_match.group(0))
    return json.loads(raw)


def detect_and_plan(
    image_rgb: np.ndarray,
    model,
    data,
    camera_name: str = "primary",
    render_size: tuple[int, int] = (342, 256),
    vla_size: tuple[int, int] = (256, 256),
) -> list[dict]:
    """Run two-prompt Gemini ER pipeline: detect objects, then generate pick-and-place plan.
    This is called once at startup (no replanning atm)."""
    image_bytes = _encode_image(image_rgb)
    h, w = image_rgb.shape[:2]

    # Prompt 1: detect red block + blue target
    prompt1 = """Locate and point to the red block and the blue target. The label returned
should be an identifying name for the object detected.
The answer should follow the json format: [{"point": <point>,
"label": <label1>}, ...]. The points are in [y, x] format
normalized to 0-1000."""

    print("  [Gemini ER] Prompt 1: detecting objects...")
    resp1 = _gemini_call(image_bytes, prompt1)
    print(f"  Response: {resp1.strip()}")
    detections = _parse_json(resp1)

    # Extract block and target coordinates
    block_det = next((d for d in detections if "block" in d["label"].lower()), detections[0])
    target_det = next((d for d in detections if "target" in d["label"].lower()), detections[-1])
    block_y, block_x = block_det["point"]
    target_y, target_x = target_det["point"]
    print(f"  Block: ({block_x}, {block_y})  Target: ({target_x}, {target_y})")

    # Prompt 2: generate pick-and-place plan
    prompt2 = f"""You are a robotic arm with six degrees-of-freedom. You have the
following functions available to you:

def move(x, y, high):
  # moves the arm to the given coordinates. The boolean value 'high' set
  to True means the robot arm should be lifted above the scene for
  avoiding obstacles during motion. 'high' set to False means the robot
  arm should have the gripper placed on the surface for interacting with
  objects.

def setGripperState(opened):
  # Opens the gripper if opened set to true, otherwise closes the gripper

Perform a pick and place operation where you pick up the block at
normalized coordinates ({block_x}, {block_y}) and place it on the
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
a high position."""

    print("  [Gemini ER] Prompt 2: generating plan...")
    resp2 = _gemini_call(image_bytes, prompt2)
    print(f"  Response: {resp2.strip()}")
    plan_steps = _parse_json(resp2)

    print(f"  Plan has {len(plan_steps)} steps")
    for i, step in enumerate(plan_steps):
        print(f"    [{i}] {step['function']}({', '.join(str(a) for a in step['args'])})")

    return plan_steps


def plan_to_waypoints(
    plan_steps: list[dict],
    model,
    data,
    camera_name: str = "primary",
    render_size: tuple[int, int] = (342, 256),
    vla_size: tuple[int, int] = (256, 256),
) -> list[tuple[np.ndarray, float]]:
    """Convert Gemini ER plan steps into a list of (xyz_3d, gripper_value) waypoints. Called once at startup (no replanning atm)."""
    vla_w, vla_h = vla_size
    gripper = 1.0  # start open
    waypoints: list[tuple[np.ndarray, float]] = []

    for step in plan_steps:
        func = step["function"]
        args = step["args"]

        if func == "move" and len(args) >= 3:
            norm_x, norm_y, high = float(args[0]), float(args[1]), bool(args[2])
            # Convert normalized 0-1000 coords to VLA pixel coords
            px_x = norm_x / 1000 * vla_w
            px_y = norm_y / 1000 * vla_h
            world_xy = pixel_to_world_3d(
                (px_x, px_y), model, data, camera_name,
                render_size=render_size, vla_size=vla_size, table_z=TABLE_Z,
            )
            if high:
                world_xy[2] = TABLE_Z + HEIGHT_OFFSET
            else:
                world_xy[2] = GRASP_HEIGHT
            waypoints.append((world_xy, gripper))

        elif func == "setGripperState" and len(args) >= 1:
            new_gripper = 1.0 if args[0] else 0.0
            if new_gripper != gripper:
                gripper = new_gripper
                # Attach gripper change to a stationary waypoint at current position
                if waypoints:
                    last_xyz = waypoints[-1][0].copy()
                    waypoints.append((last_xyz, gripper))

    return waypoints


def visualize_waypoints(
    viewer,
    waypoints: list[tuple[np.ndarray, float]],
    current_idx: int = 0,
):
    """Draw waypoint markers in the viewer. Current waypoint is bright, others are dim."""
    idx = 0
    for i, (wp_xyz, wp_grip) in enumerate(waypoints):
        is_current = (i == current_idx)
        # Green=open gripper, red=closed; bright if current, dim if not
        alpha = 0.9 if is_current else 0.3
        size = 0.012 if is_current else 0.008
        if wp_grip > 0.5:
            rgba = np.array([0.0, 1.0, 0.0, alpha], dtype=np.float32)  # green = open
        else:
            rgba = np.array([1.0, 0.0, 0.0, alpha], dtype=np.float32)  # red = closed
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[idx],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[size, 0, 0],
            pos=wp_xyz.astype(np.float64),
            mat=np.eye(3).flatten(),
            rgba=rgba,
        )
        idx += 1
    viewer.user_scn.ngeom = idx
