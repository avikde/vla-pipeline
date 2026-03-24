"""Gemini ER object detection + MuJoCo camera calibration for pixel-to-3D projection."""

import io
import json
import re

import numpy as np
from google import genai
from google.genai import types
from PIL import Image


def detect_block_pixel(
    image_rgb: np.ndarray, label: str = "red block"
) -> tuple[float, float] | None:
    """Send image to Gemini ER, return (x_px, y_px) in image coords for the best-matching label."""
    h, w = image_rgb.shape[:2]

    prompt = """
        Point to no more than 10 items in the image. The label returned
        should be an identifying name for the object detected.
        The answer should follow the json format: [{"point": <point>,
        "label": <label1>}, ...]. The points are in [y, x] format
        normalized to 0-1000.
    """
    client = genai.Client()

    # Encode as PNG bytes
    img_pil = Image.fromarray(image_rgb)
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    image_bytes = buf.getvalue()

    response = client.models.generate_content(
        model="gemini-robotics-er-1.5-preview",
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            prompt,
        ],
        config=types.GenerateContentConfig(
            temperature=0.5,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )

    # Parse JSON (strip markdown fences if present)
    raw = response.text.strip()
    print(f"  Gemini ER raw response: {raw}")
    json_match = re.search(r"```json?\s*(.*?)\s*```", raw, re.DOTALL)
    detections = json.loads(json_match.group(1) if json_match else raw)

    # Find best matching label (case-insensitive substring match)
    label_lower = label.lower()
    best = None
    for det in detections:
        det_label = det["label"].lower()
        if label_lower in det_label or det_label in label_lower:
            best = det
            break
    if best is None and detections:
        print(f"  ⚠️  No detection matched '{label}', using first: '{detections[0]['label']}'")
        best = detections[0]
    if best is None:
        print(f"  ❌ No detections returned by Gemini ER")
        return None

    y_norm, x_norm = best["point"]
    x_px = x_norm / 1000 * w
    y_px = y_norm / 1000 * h
    print(f"  Gemini ER detected '{best['label']}' at pixel ({x_px:.1f}, {y_px:.1f})")
    return (x_px, y_px)


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
