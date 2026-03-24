#!/usr/bin/env python3
"""
X-VLA with WidowX Robot Demo - DEBUG VERSION

Run with --no-vla to skip model loading and use a reference policy instead.
Run with --verbose to print per-step action/alignment info to stdout.
Run with --step-size 0.005 to slow the reference policy.
Run with --kp 10 to lower arm position control gains.
"""

import argparse
import mujoco
import mujoco.viewer as mj_viewer
import numpy as np
import torch
from PIL import Image, ImageDraw
import sys

import xvla_policy as xvla
import widowx_control as ctrl

# Parse arguments
parser = argparse.ArgumentParser(description='X-VLA with WidowX Robot - Debug')
parser.add_argument('--verbose', '-v', action='store_true', help='Print per-step action/alignment info')
parser.add_argument('--no-vla', '-n', action='store_true', help='Skip VLA loading; use reference policy instead')
parser.add_argument('--dry-run', '-d', action='store_true',
                    help='Visualize trajectory dots only; skip IK and control application')
parser.add_argument('--step-size', type=float, default=0.02,
                    help='EE step size per timestep for reference policy (m). Default=0.02')
parser.add_argument('--kp', type=float, default=None,
                    help='Override arm joint position gain kp (default: model value ~50)')
parser.add_argument('--up', action='store_true',
                    help='Reference policy: target 0.2m above initial EE (ignore cube); tests IK/control in isolation')
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

    if args.kp is not None:
        for i in range(6):  # arm joints only (skip gripper at index 6)
            model.actuator_gainprm[i, 0] = args.kp
            model.actuator_biasprm[i, 1] = -args.kp
        print(f"  ✓ Arm kp overridden to {args.kp}")

    for i in range(model.nu):
        print(f"     [{i}] {model.actuator(i).name}: kp={model.actuator_gainprm[i, 0]:.1f}  range={model.actuator_ctrlrange[i]} limited={model.actuator_ctrllimited[i]}")

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
    print("  ⚠️  No VLA — language tokens skipped")

# Settle physics
print("\n[5/7] Settling physics...")
for _ in range(100):
    mujoco.mj_step(model, data)
print("  ✓ Physics settled")

# Get initial state
initial_ee_pos, _ = xvla.get_ee_pose(model, data)
cube_pos = xvla.get_cube_position(model, data)
initial_ee_state = xvla.get_ee_state_8d(model, data)

print("\n  📍 Initial State:")
print(f"     - EE position: [{initial_ee_pos[0]:.3f}, {initial_ee_pos[1]:.3f}, {initial_ee_pos[2]:.3f}]")
print(f"     - EE state (8D): [x={initial_ee_state[0]:.3f}, y={initial_ee_state[1]:.3f}, z={initial_ee_state[2]:.3f}, "
      f"roll={initial_ee_state[3]:.3f}, pitch={initial_ee_state[4]:.3f}, yaw={initial_ee_state[5]:.3f}, "
      f"pad={initial_ee_state[6]:.3f}, gripper={initial_ee_state[7]:.3f}]")
if cube_pos is not None:
    print(f"     - Cube position: [{cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f}]")
    print(f"     - Distance to cube: {np.linalg.norm(initial_ee_pos - cube_pos):.3f}m")

# Build controller (allocates scratch MjData + caches IDs once).
# Orientation IK disabled: the no-VLA reference keeps current Euler angles which
# causes wrist_rotate to spin through singularities. Position-only IK is sufficient
# for validating reach behaviour.
controller = ctrl.WidowXController(model, use_orientation=False)
# Home arm joints used as fixed null-space reference to prevent IK from drifting
# into a drooped configuration branch.
home_null_ref = model.keyframe('home').ctrl[:6].copy()

# Launch viewer
print("\n[6/7] Launching viewer...")
viewer = mj_viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=args.dry_run)
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
smoothed_gripper = 1.0  # start open; EMA-filtered to suppress VLA gripper jitter

while True:
    # 1. Render cameras
    #    Only use the 'up' (over-shoulder) view for the VLA — BridgeData's
    #    second camera is often black, so pass a black image for image2.
    img = render_camera('up')
    img2 = np.zeros((VLA_HEIGHT, VLA_WIDTH, 3), dtype=np.uint8)

    # 2. Build observation and run inference
    actions_np = None
    ee_state_8d = xvla.get_ee_state_8d(model, data)
    cube_pos_now = xvla.get_cube_position(model, data)

    if policy is not None:
        observation = xvla.build_observation(
            img, img2, ee_state_8d, language_tokens, language_attention_mask, device # pyright:ignore
        )
        actions_np = xvla.select_action(policy, observation, device)

        action_queue = policy._queues.get("action", [])
        queue_size = len(action_queue)
        is_new_chunk = queue_size == policy.config.chunk_size - 1
        if is_new_chunk:
            cached_action_targets = []
            for queued_action in list(action_queue)[:NUM_MARKERS]:
                if isinstance(queued_action, torch.Tensor):
                    cached_action_targets.append(queued_action.flatten()[:10].cpu().numpy())
                else:
                    cached_action_targets.append(np.array(queued_action).flatten()[:10])
    else:
        queue_size = 0
        is_new_chunk = True
        current_xyz = ee_state_8d[0:3]
        rot6d = xvla.rotation_matrix_to_6d(xvla.euler_to_rotation_matrix(*ee_state_8d[3:6]))
        if args.up:
            # Fixed target 0.2m above initial EE; ignores cube and current orientation drift.
            goal_xyz = initial_ee_pos + np.array([0.0, 0.0, 0.2])
            goal_rot6d = xvla.rotation_matrix_to_6d(np.eye(3))
        else:
            block_pos = cube_pos_now if cube_pos_now is not None else np.zeros(3)
            actions_np = xvla.generate_non_vla_reference(ee_state_8d, [img, img2], block_pos, step_size=args.step_size)
            goal_xyz = block_pos
            goal_rot6d = rot6d
        cached_action_targets = []
        for t in np.linspace(1 / NUM_MARKERS, 1, NUM_MARKERS):
            wp = np.zeros(10, dtype=np.float32)
            wp[0:3] = current_xyz + (goal_xyz - current_xyz) * t
            wp[3:9] = goal_rot6d
            wp[9]   = 1.0  # gripper open
            cached_action_targets.append(wp)

    # 3. Per-step debug output (--verbose only)
    if args.verbose:
        current_ee_pos, current_ee_rot = xvla.get_ee_pose(model, data)
        xyz_1, rot6d_1, gripper_1 = xvla.decode_ee6d_action(actions_np) if actions_np is not None else (None, None, 0.0)

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

    # 4. IK + control (skipped in --dry-run mode)
    #    Use the current dequeued action (actions_np) for IK each step,
    #    not the stale cached first entry. cached_action_targets is for viz only.
    ik_target_pos = None
    current_action_10d = None
    if actions_np is not None and len(actions_np) >= 10:
        current_action_10d = actions_np[:10].copy()
    if not args.dry_run and current_action_10d is not None:
        ik_target_pos = current_action_10d[:3].copy()
        # Smooth gripper to suppress VLA output jitter
        GRIPPER_EMA_ALPHA = 0.15
        smoothed_gripper += GRIPPER_EMA_ALPHA * (float(current_action_10d[9]) - smoothed_gripper)
        current_action_10d[9] = smoothed_gripper
        ctrl_target = controller.solve_ik(data.qpos, [current_action_10d], debug=args.verbose)
        if ctrl_target is not None:
            controller.apply_control(data.ctrl, ctrl_target)
            if args.verbose:
                print(f"  ctrl_target: joints={fmt_vec(ctrl_target[:6])}  gripper={fmt(ctrl_target[6])}")
                print(f"  data.ctrl:   joints={fmt_vec(data.ctrl[:6])}  gripper={fmt(data.ctrl[6])}")
                print(f"  qpos vs ctrl: {fmt_vec(data.qpos[:6] - data.ctrl[:6])}")

    # Step simulation
    mujoco.mj_step(model, data)

    # Post-step: compare actual EE vs IK target to diagnose servo tracking
    if args.verbose and ik_target_pos is not None:
        actual_ee_pos, _ = xvla.get_ee_pose(model, data)
        servo_err = np.linalg.norm(actual_ee_pos - ik_target_pos)
        print(f"  Post-step EE: {fmt_vec(actual_ee_pos)}  (target: {fmt_vec(ik_target_pos)})  servo_err={fmt(servo_err)}m")
        print(f"  qpos:  {fmt_vec(data.qpos[:6])}")
        print(f"  ctrl:  {fmt_vec(data.ctrl[:6])}")

    # 5. Viewer sync + trajectory dots (xyz only for rendering)
    viewer.user_scn.ngeom = 0
    for i, target in enumerate(cached_action_targets):
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[i],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.01, 0, 0],
            pos=target[:3].astype(np.float64),
            mat=np.eye(3).flatten(),
            rgba=MARKER_COLORS[i],
        )
    viewer.user_scn.ngeom = len(cached_action_targets)

    # Save camera snapshot once after first trajectory is available
    if not camera_snapshot_saved and cached_action_targets:
        W, H = VLA_WIDTH, VLA_HEIGHT
        traj = [t[:3] for t in cached_action_targets]
        snap_up = render_camera('up', trajectory=traj)
        snap_side = render_camera('side', trajectory=traj)
        combined = Image.new('RGB', (W * 2 + 20, H + 30), color=(255, 255, 255))
        combined.paste(Image.fromarray(snap_up), (0, 30))
        combined.paste(Image.fromarray(snap_side), (W + 20, 30))
        draw = ImageDraw.Draw(combined)
        draw.text((W // 2 - 50, 5), 'image (VLA input)', fill=(0, 0, 0))
        draw.text((W + 20 + W // 2 - 30, 5), 'side (debug only)', fill=(0, 0, 0))
        combined.save('camera_views.png')
        print("  Saved camera snapshot → camera_views.png")
        camera_snapshot_saved = True

    viewer.sync()
    if not viewer.is_running():
        print(f"\nViewer closed at step {step}")
        break

    step += 1

viewer.close()
print("\nDemo complete!")
