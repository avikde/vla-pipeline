#!/usr/bin/env python3
"""
X-VLA with WidowX Robot Demo - DEBUG VERSION

Run with --no-vla to skip model loading and use a reference policy instead.
Run with --verbose to print per-step action/alignment info to stdout.
"""

import argparse
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw
import sys

import xvla_policy as xvla

# Parse arguments
parser = argparse.ArgumentParser(description='X-VLA with WidowX Robot - Debug')
parser.add_argument('--verbose', action='store_true', help='Print per-step action/alignment info')
parser.add_argument('--no-vla', action='store_true', help='Skip VLA loading; use reference policy instead')
args = parser.parse_args()

# Precompute trajectory marker colors (green -> red gradient, 10 markers)
NUM_MARKERS = 10
MARKER_COLORS = []
for i in range(NUM_MARKERS):
    fade = i / max(1, NUM_MARKERS - 1)
    MARKER_COLORS.append(np.array([fade, 1.0 - fade, 0.0, 0.6], dtype=np.float32))

# Load X-VLA policy (skipped in --no-vla mode)
policy = tokenizer = language_tokens = language_attention_mask = None
device = "cpu"

print("=" * 60)
print("X-VLA WidowX Demo - DEBUG MODE")
print("=" * 60)

if not args.no_vla:
    print("\n[1/7] Loading X-VLA WidowX policy...")
    try:
        policy, tokenizer, device = xvla.load_policy("lerobot/xvla-widowx")
    except ImportError as e:
        print(f"\n❌ X-VLA not installed: {e}")
        print('  pip install "lerobot[xvla]"')
        sys.exit(1)
else:
    print("\n[1/7] Skipping VLA load (--no-vla). Using reference policy.")

# Load WidowX MuJoCo model
print("\n[2/7] Loading WidowX MuJoCo model...")
try:
    xml_path = 'assets/widowx/widowx_vision_scene.xml'
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    home_qpos = model.keyframe('home').qpos
    n_robot_joints = 8  # 6 arm + 2 finger
    data.qpos[:n_robot_joints] = home_qpos[:n_robot_joints]
    data.ctrl[:] = model.keyframe('home').ctrl
    mujoco.mj_forward(model, data)

    print(f"  ✓ Model loaded (initialized to 'home' keyframe)")
    print(f"     - nq={model.nq}  nv={model.nv}  nu={model.nu}")

    print(f"\n  🎮 Actuator Control Modes:")
    for i in range(model.nu):
        print(f"     [{i}] {model.actuator(i).name}: range={model.actuator_ctrlrange[i]} limited={model.actuator_ctrllimited[i]}")

except Exception as e:
    print(f"❌ Error loading WidowX model: {e}")
    sys.exit(1)

# Setup renderer
print("\n[3/7] Setting up renderer...")
VLA_WIDTH, VLA_HEIGHT = 256, 256
model.vis.global_.offwidth = max(model.vis.global_.offwidth, VLA_WIDTH)
model.vis.global_.offheight = max(model.vis.global_.offheight, VLA_HEIGHT)
renderer = mujoco.Renderer(model, height=VLA_HEIGHT, width=VLA_WIDTH)
print(f"  ✓ Renderer ready ({VLA_WIDTH}x{VLA_HEIGHT})")

def render_camera(camera_name, trajectory=None):
    """Render from a specific camera, optionally with trajectory spheres."""
    camera_id = model.camera(camera_name).id
    renderer.update_scene(data, camera=camera_id)
    if trajectory:
        for i, target_xyz in enumerate(trajectory):
            mujoco.mjv_initGeom(
                renderer.scene.geoms[renderer.scene.ngeom],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.01, 0, 0],
                pos=target_xyz.astype(np.float64),
                mat=np.eye(3).flatten(),
                rgba=MARKER_COLORS[i],
            )
            renderer.scene.ngeom += 1
    return renderer.render()


# Task instruction
print("\n[4/7] Setting up task...")
task_instruction = "Pick up the red block"
print(f"  Task: '{task_instruction}'")

if policy is not None:
    language_tokens, language_attention_mask = xvla.tokenize_task(
        task_instruction, tokenizer, policy.config, device
    )
    print(f"  ✓ Language tokens: {language_tokens.shape}")
else:
    print(f"  ⚠️  No VLA — language tokens skipped")

# Settle physics
print("\n[5/7] Settling physics...")
for _ in range(100):
    mujoco.mj_step(model, data)
print(f"  ✓ Physics settled")

# Get initial state
initial_ee_pos, _ = xvla.get_ee_pose(model, data)
cube_pos = xvla.get_cube_position(model, data)
initial_ee_state = xvla.get_ee_state_8d(model, data)

print(f"\n  📍 Initial State:")
print(f"     - EE position: [{initial_ee_pos[0]:.3f}, {initial_ee_pos[1]:.3f}, {initial_ee_pos[2]:.3f}]")
print(f"     - EE state (8D): [x={initial_ee_state[0]:.3f}, y={initial_ee_state[1]:.3f}, z={initial_ee_state[2]:.3f}, "
      f"roll={initial_ee_state[3]:.3f}, pitch={initial_ee_state[4]:.3f}, yaw={initial_ee_state[5]:.3f}, "
      f"pad={initial_ee_state[6]:.3f}, gripper={initial_ee_state[7]:.3f}]")
if cube_pos is not None:
    print(f"     - Cube position: [{cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f}]")
    print(f"     - Distance to cube: {np.linalg.norm(initial_ee_pos - cube_pos):.3f}m")

# Launch viewer
print("\n[6/7] Launching viewer...")
import mujoco.viewer as mj_viewer
viewer = mj_viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=True)
viewer.cam.distance = 0.8
viewer.cam.azimuth = 45
viewer.cam.elevation = -20
viewer.cam.lookat[:] = [0.2, 0.0, 0.2]
print(f"  ✓ Viewer launched  |  --verbose={'on' if args.verbose else 'off'}  |  Close window to stop.")

# Simulation loop
print("\n[7/7] Running inference loop...")
print("=" * 60)

def fmt(v):
    return f"{v:.4f}"

def fmt_vec(v):
    return "[" + ", ".join(f"{x:.4f}" for x in v) + "]"

step = 0
cached_action_targets = []
camera_snapshot_saved = False

while True:
    # 1. Render cameras
    img = render_camera('up')
    img2 = render_camera('side')

    # 2. Build observation and run inference
    ee_state_8d = xvla.get_ee_state_8d(model, data)
    cube_pos_now = xvla.get_cube_position(model, data)

    if policy is not None:
        observation = xvla.build_observation(
            img, img2, ee_state_8d, language_tokens, language_attention_mask, device
        )
        actions_np = xvla.select_action(policy, observation, device)

        action_queue = policy._queues.get("action", [])
        queue_size = len(action_queue)
        is_new_chunk = queue_size == policy.config.chunk_size - 1
        if is_new_chunk:
            cached_action_targets = []
            for queued_action in list(action_queue)[:NUM_MARKERS]:
                if isinstance(queued_action, torch.Tensor):
                    cached_action_targets.append(queued_action.flatten()[:3].cpu().numpy())
                else:
                    cached_action_targets.append(np.array(queued_action).flatten()[:3])
    else:
        block_pos = cube_pos_now if cube_pos_now is not None else np.zeros(3)
        actions_np = xvla.generate_non_vla_reference(ee_state_8d, [img, img2], block_pos)
        queue_size = 0
        is_new_chunk = True
        current_xyz = ee_state_8d[0:3]
        cached_action_targets = [
            current_xyz + (block_pos - current_xyz) * t
            for t in np.linspace(0, 1, NUM_MARKERS)
        ]

    # 3. Per-step debug output (--verbose only)
    if args.verbose:
        current_ee_pos, current_ee_rot = xvla.get_ee_pose(model, data)
        xyz_1, rot6d_1, gripper_1 = xvla.decode_ee6d_action(actions_np)

        print(f"\n--- Step {step} {'(NEW CHUNK) ' if is_new_chunk else ''}queue={queue_size} ---")
        print(f"  Proprio:    {fmt_vec(ee_state_8d)}")
        if xyz_1 is not None:
            print(f"  Target XYZ: {fmt_vec(xyz_1)}  gripper={fmt(gripper_1)}")
            action_delta = xyz_1 - current_ee_pos
            if cube_pos_now is not None:
                cube_dir = cube_pos_now - current_ee_pos
                dn, cd = np.linalg.norm(action_delta), np.linalg.norm(cube_dir)
                alignment = np.dot(action_delta / (dn + 1e-8), cube_dir / (cd + 1e-8))
                label = "toward cube" if alignment > 0.3 else ("AWAY from cube" if alignment < -0.3 else "orthogonal")
                print(f"  Alignment:  {fmt(alignment)} ({label})  dist_to_cube={fmt(cd)}m")

    # 4. Step simulation
    mujoco.mj_step(model, data)

    # 5. Viewer sync + trajectory dots
    viewer.user_scn.ngeom = 0
    for i, target_xyz in enumerate(cached_action_targets):
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[i],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.01, 0, 0],
            pos=target_xyz.astype(np.float64),
            mat=np.eye(3).flatten(),
            rgba=MARKER_COLORS[i],
        )
    viewer.user_scn.ngeom = len(cached_action_targets)

    # Save camera snapshot once after first trajectory is available
    if not camera_snapshot_saved and cached_action_targets:
        W, H = VLA_WIDTH, VLA_HEIGHT
        traj = cached_action_targets
        snap_up = render_camera('up', trajectory=traj)
        snap_side = render_camera('side', trajectory=traj)
        combined = Image.new('RGB', (W * 2 + 20, H + 30), color=(255, 255, 255))
        combined.paste(Image.fromarray(snap_up), (0, 30))
        combined.paste(Image.fromarray(snap_side), (W + 20, 30))
        draw = ImageDraw.Draw(combined)
        draw.text((W // 2 - 50, 5), 'image (over the shoulder)', fill=(0, 0, 0))
        draw.text((W + 20 + W // 2 - 30, 5), 'image2 (side)', fill=(0, 0, 0))
        combined.save('camera_views.png')
        print(f"  Saved camera snapshot → camera_views.png")
        camera_snapshot_saved = True

    viewer.sync()
    if not viewer.is_running():
        print(f"\nViewer closed at step {step}")
        break

    step += 1

viewer.close()
print("\nDemo complete!")
