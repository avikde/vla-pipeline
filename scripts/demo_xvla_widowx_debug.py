#!/usr/bin/env python3
"""
X-VLA with WidowX Robot Demo - DEBUG VERSION

Enhanced with detailed debugging for action space issues:
1. Verifies X-VLA is configured for EE actions (ee6d mode)
2. Logs raw action vectors (all 20 dimensions)
3. Extracts and logs EE poses (current and target)
4. Tracks cube position and checks if actions point toward it
5. Verifies normalization/unnormalization
6. Checks control mode (position/velocity/torque)
"""

import argparse
import mujoco
import numpy as np
import torch
import sys

# Parse arguments
parser = argparse.ArgumentParser(description='X-VLA with WidowX Robot - Debug')
parser.add_argument('--verbose', action='store_true', help='Print detailed action info every step')
args = parser.parse_args()

print("=" * 60)
print("X-VLA WidowX Demo - DEBUG MODE")
print("=" * 60)

# Helper functions (defined before use)
def add_trajectory_sites_to_xml(xml_path, output_path, num_sites=10):
    """
    Add visualization sites to the MuJoCo XML for trajectory markers.
    This modifies the XML file to include mocap bodies for visualization.
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Find or create worldbody
    worldbody = root.find('worldbody')
    if worldbody is None:
        worldbody = ET.SubElement(root, 'worldbody')

    # Add mocap bodies for trajectory visualization
    for i in range(num_sites):
        mocap = ET.SubElement(worldbody, 'body', {
            'name': f'traj_marker_{i}',
            'mocap': 'true',
            'pos': '0 0 -10'  # Start hidden underground
        })
        # Add a small sphere geom
        fade = i / max(1, num_sites - 1)
        rgba = f"{fade} {1-fade} 0 0.6"  # Green→Red gradient
        ET.SubElement(mocap, 'geom', {
            'type': 'sphere',
            'size': '0.01',
            'rgba': rgba,
            'contype': '0',
            'conaffinity': '0',
            'group': '1'
        })

    tree.write(output_path)
    print(f"  📝 Created modified scene with trajectory markers: {output_path}")
    return output_path

# Load X-VLA policy
print("\n[1/7] Loading X-VLA WidowX policy...")
try:
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
    from transformers import AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  ✓ Device: {device}")

    # Load WidowX-specific checkpoint
    policy = XVLAPolicy.from_pretrained("lerobot/xvla-widowx").to(device).eval()

    # Load language tokenizer (X-VLA uses BART)
    tokenizer = AutoTokenizer.from_pretrained(policy.config.tokenizer_name)

    # ===== DEBUG: Check policy configuration =====
    print(f"\n  📋 Policy Configuration:")
    print(f"     - action_mode: {policy.config.action_mode}")
    print(f"     - chunk_size: {policy.config.chunk_size}")
    print(f"     - n_action_steps: {policy.config.n_action_steps}")
    print(f"     - num_denoising_steps: {policy.config.num_denoising_steps}")
    print(f"     - use_proprio: {policy.config.use_proprio}")
    print(f"     - max_action_dim: {policy.config.max_action_dim}")

    if policy.config.action_mode != "ee6d":
        print(f"\n  ⚠️  WARNING: action_mode is '{policy.config.action_mode}', expected 'ee6d'")
        print(f"     X-VLA WidowX is trained with EE actions, not joint actions!")
    else:
        print(f"  ✓ Confirmed: Using EE (end-effector) action mode")

except ImportError as e:
    print(f"\n❌ X-VLA not installed: {e}")
    print("\nTo install X-VLA:")
    print('  pip install "lerobot[xvla]"')
    sys.exit(1)

# Load WidowX MuJoCo model
print("\n[2/7] Loading WidowX MuJoCo model...")
try:
    # Create modified XML with trajectory markers
    xml_path = 'assets/widowx/widowx_vision_scene.xml'
    modified_xml = 'assets/widowx/widowx_vision_scene_with_markers.xml'
    add_trajectory_sites_to_xml(xml_path, modified_xml, num_sites=10)
    xml_path = modified_xml

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    print(f"  ✓ Model loaded")
    print(f"     - nq (positions): {model.nq}")
    print(f"     - nv (velocities): {model.nv}")
    print(f"     - nu (actuators): {model.nu}")

    # Check actuator control mode
    print(f"\n  🎮 Actuator Control Modes:")
    for i in range(model.nu):
        actuator_name = model.actuator(i).name
        # dyntype: 0=None, 1=Integrator, 2=Filter, 3=Muscle, 4=User
        # gaintype: 0=Fixed, 1=Affine, 2=Muscle, 3=User
        # biastype: 0=None, 1=Affine, 2=Muscle, 3=User
        ctrl_limited = model.actuator_ctrllimited[i]
        ctrl_range = model.actuator_ctrlrange[i]
        print(f"     [{i}] {actuator_name}: range={ctrl_range} limited={ctrl_limited}")

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

def render_camera(camera_name):
    """Render from a specific camera."""
    camera_id = model.camera(camera_name).id
    renderer.update_scene(data, camera=camera_id)
    return renderer.render()

def preprocess_image(rgb_image, device='cpu'):
    """Preprocess image for VLA input."""
    img_tensor = torch.from_numpy(rgb_image).permute(2, 0, 1).float() / 255.0
    return img_tensor.unsqueeze(0).to(device)

def get_ee_pose(model, data):
    """Get end-effector pose (position + rotation matrix)."""
    # WidowX end-effector is the "wx250s/gripper_link" body
    ee_body_id = model.body("wx250s/gripper_link").id
    ee_pos = data.xpos[ee_body_id].copy()
    ee_rot = data.xmat[ee_body_id].reshape(3, 3).copy()
    return ee_pos, ee_rot

def rotation_matrix_to_6d(rot_mat):
    """Convert 3x3 rotation matrix to 6D representation (first two columns)."""
    return rot_mat[:, :2].flatten()

def rotation_matrix_to_euler(rot_mat):
    """Convert 3x3 rotation matrix to roll, pitch, yaw (XYZ Euler angles).

    This matches the BridgeData convention for observation.state.
    """
    # Extract pitch from R[2,0]
    sy = np.sqrt(rot_mat[0, 0]**2 + rot_mat[1, 0]**2)
    singular = sy < 1e-6
    if not singular:
        roll = np.arctan2(rot_mat[2, 1], rot_mat[2, 2])
        pitch = np.arctan2(-rot_mat[2, 0], sy)
        yaw = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
    else:
        roll = np.arctan2(-rot_mat[1, 2], rot_mat[1, 1])
        pitch = np.arctan2(-rot_mat[2, 0], sy)
        yaw = 0.0
    return roll, pitch, yaw

def get_ee_state_8d(model, data):
    """Get the 8D end-effector state matching BridgeData observation.state format.

    Returns: np.array of shape (8,) with [x, y, z, roll, pitch, yaw, pad, gripper]

    BridgeData convention (from IPEC-COMMUNITY/bridge_orig_lerobot):
      [0] x       - EE position X
      [1] y       - EE position Y
      [2] z       - EE position Z
      [3] roll    - EE rotation roll
      [4] pitch   - EE rotation pitch
      [5] yaw     - EE rotation yaw
      [6] pad     - Always 0
      [7] gripper - Gripper openness (left_finger joint position)
    """
    ee_pos, ee_rot = get_ee_pose(model, data)
    roll, pitch, yaw = rotation_matrix_to_euler(ee_rot)

    # Gripper state: left_finger joint position
    gripper_joint_id = model.joint("left_finger").id
    gripper_pos = data.qpos[gripper_joint_id]

    return np.array([
        ee_pos[0], ee_pos[1], ee_pos[2],
        roll, pitch, yaw,
        0.0,          # pad
        gripper_pos,  # gripper
    ], dtype=np.float32)

def get_cube_position(model, data, cube_name="red_block"):
    """Get position of the target cube."""
    try:
        cube_body_id = model.body(cube_name).id
        return data.xpos[cube_body_id].copy()
    except:
        # Try alternative names
        for alt_name in ["blue_block", "red_box", "cube"]:
            try:
                cube_body_id = model.body(alt_name).id
                return data.xpos[cube_body_id].copy()
            except:
                continue
        return None

def decode_ee6d_action(action_vec):
    """
    Decode the 20D EE action vector.

    Based on lerobot action_hub.py:
    - Indices 0-2, 10-12: XYZ position (duplicate for temporal sequence)
    - Indices 3-8, 13-18: 6D rotation (duplicate for temporal sequence)
    - Indices 9, 19: Gripper (duplicate for temporal sequence)
    """
    if len(action_vec) < 20:
        print(f"  ⚠️  Action vector too short: {len(action_vec)}, expected 20")
        return None, None, None, None

    # Extract first timestep (indices 0-9)
    xyz_1 = action_vec[0:3]
    rot6d_1 = action_vec[3:9]
    gripper_1 = action_vec[9]

    # Extract second timestep (indices 10-19)
    xyz_2 = action_vec[10:13]
    rot6d_2 = action_vec[13:19]
    gripper_2 = action_vec[19]

    return xyz_1, rot6d_1, gripper_1, xyz_2, rot6d_2, gripper_2

# Task instruction
print("\n[4/7] Setting up task...")
task_instruction = "Pick up the red block"
print(f"  Task: '{task_instruction}'")

# Tokenize language instruction once (before the loop)
tokenized = tokenizer(
    task_instruction,
    padding='max_length',
    max_length=policy.config.tokenizer_max_length,
    truncation=True,
    return_tensors='pt'
)
language_tokens = tokenized['input_ids'].to(device)
language_attention_mask = tokenized['attention_mask'].to(device)
print(f"  ✓ Language tokens: {language_tokens.shape}")

# Settle physics
print("\n[5/7] Settling physics...")
for _ in range(100):
    mujoco.mj_step(model, data)
print(f"  ✓ Physics settled")

# Get initial state
initial_ee_pos, initial_ee_rot = get_ee_pose(model, data)
cube_pos = get_cube_position(model, data)

initial_ee_state = get_ee_state_8d(model, data)

print(f"\n  📍 Initial State:")
print(f"     - EE position: [{initial_ee_pos[0]:.3f}, {initial_ee_pos[1]:.3f}, {initial_ee_pos[2]:.3f}]")
print(f"     - EE state (8D): {initial_ee_state}")
print(f"       [x={initial_ee_state[0]:.3f}, y={initial_ee_state[1]:.3f}, z={initial_ee_state[2]:.3f}, "
      f"roll={initial_ee_state[3]:.3f}, pitch={initial_ee_state[4]:.3f}, yaw={initial_ee_state[5]:.3f}, "
      f"pad={initial_ee_state[6]:.3f}, gripper={initial_ee_state[7]:.3f}]")
if cube_pos is not None:
    print(f"     - Cube position: [{cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f}]")
    dist_to_cube = np.linalg.norm(initial_ee_pos - cube_pos)
    print(f"     - Distance to cube: {dist_to_cube:.3f}m")

# Launch viewer
print("\n[6/7] Launching viewer...")
import mujoco.viewer as mj_viewer
viewer = mj_viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=True)
viewer.cam.distance = 0.8
viewer.cam.azimuth = 45
viewer.cam.elevation = -20
viewer.cam.lookat[:] = [0.2, 0.0, 0.2]
print(f"  ✓ Viewer launched (manual control via right panel sliders)")
print(f"  🎯 Trajectory visualization enabled")
print(f"  Close the viewer window to stop.")

# Simulation loop
print("\n[7/7] Running X-VLA inference loop...")
print("=" * 60)

# Create log file for detailed action analysis
log_file = open("xvla_action_debug.log", "w", encoding="utf-8")
log_file.write("X-VLA Action Debug Log\n")
log_file.write("=" * 80 + "\n")
log_file.write(f"Task: {task_instruction}\n")
log_file.write(f"Action mode: {policy.config.action_mode}\n")
log_file.write("=" * 80 + "\n\n")

# Track previous action's timestep_2 to check continuity
prev_xyz_2 = None
step = 0

while True:
    # 1. Render cameras (matching X-VLA WidowX training: "up" and "side")
    try:
        img_up = render_camera('up')
        img_side = render_camera('side')
    except:
        img_up = render_camera('third_person')
        img_side = img_up

    # 2. Preprocess for X-VLA
    img_up_tensor = preprocess_image(img_up, device=device)
    img_side_tensor = preprocess_image(img_side, device=device)

    # Get robot state as 8D EE state matching BridgeData format:
    # [x, y, z, roll, pitch, yaw, pad, gripper]
    ee_state_8d = get_ee_state_8d(model, data)

    # X-VLA WidowX expects specific observation keys
    observation = {
        'observation.images.image': img_up_tensor,
        'observation.images.image2': img_side_tensor,
        'observation.state': torch.from_numpy(ee_state_8d).float().unsqueeze(0).to(device),
        'observation.language.tokens': language_tokens,
        'observation.language.attention_mask': language_attention_mask,
    }

    # 3. VLA inference
    with torch.inference_mode():
        try:
            actions = policy.select_action(observation)
        except Exception as e:
            print(f"❌ VLA inference error at step {step}: {e}")
            actions = torch.zeros(20, device=device)  # EE6D is 20-dimensional

    if device == "cuda":
        torch.cuda.synchronize()

    # Convert actions to numpy
    if isinstance(actions, torch.Tensor):
        actions_np = actions.detach().cpu().numpy().flatten()
    else:
        actions_np = np.array(actions).flatten()

    # ===== DEBUG: Analyze raw actions =====
    current_ee_pos, current_ee_rot = get_ee_pose(model, data)
    cube_pos = get_cube_position(model, data)

    log_entry = f"\n{'='*80}\nStep {step}\n{'='*80}\n"
    log_entry += f"Raw action vector (len={len(actions_np)}):\n"
    log_entry += f"  {actions_np}\n\n"

    # Decode EE actions
    if len(actions_np) >= 20:
        xyz_1, rot6d_1, gripper_1, xyz_2, rot6d_2, gripper_2 = decode_ee6d_action(actions_np)

        # Check difference between timestep 1 and timestep 2
        xyz_diff = np.linalg.norm(xyz_2 - xyz_1)

        # Track chunk boundaries
        queue_size = len(policy._queues.get("action", []))
        is_new_chunk = queue_size == 31  # Just generated new chunk (32 actions, popped 1)

        # Check continuity: does prev_timestep_2 match current_timestep_1?
        continuity_gap = None
        if prev_xyz_2 is not None:
            continuity_gap = np.linalg.norm(xyz_1 - prev_xyz_2)

        if is_new_chunk:
            print(f"\n  🔄 NEW CHUNK generated at step {step}")
            if continuity_gap is not None:
                print(f"     Continuity gap (prev_t2 → curr_t1): {continuity_gap:.4f}")

        if step % 10 == 0 or args.verbose:
            print(f"  XYZ diff (t1→t2): {xyz_diff:.4f}, Queue: {queue_size}")
            if continuity_gap is not None:
                print(f"  Continuity (prev_t2→t1): {continuity_gap:.4f}")

        # Store for next iteration
        prev_xyz_2 = xyz_2.copy()

        log_entry += f"Decoded EE Actions (timestep 1):\n"
        log_entry += f"  Position (XYZ): [{xyz_1[0]:.4f}, {xyz_1[1]:.4f}, {xyz_1[2]:.4f}]\n"
        log_entry += f"  Rotation (6D):  [{rot6d_1[0]:.4f}, {rot6d_1[1]:.4f}, {rot6d_1[2]:.4f}, {rot6d_1[3]:.4f}, {rot6d_1[4]:.4f}, {rot6d_1[5]:.4f}]\n"
        log_entry += f"  Gripper:        {gripper_1:.4f}\n\n"

        log_entry += f"Decoded EE Actions (timestep 2):\n"
        log_entry += f"  Position (XYZ): [{xyz_2[0]:.4f}, {xyz_2[1]:.4f}, {xyz_2[2]:.4f}]\n"
        log_entry += f"  XYZ diff (t1→t2): {xyz_diff:.4f}\n"

        # Test delta encoding hypothesis
        xyz_sum = xyz_1 + xyz_2
        log_entry += f"\n  🔍 Delta Hypothesis Test:\n"
        log_entry += f"     t1 + t2 = [{xyz_sum[0]:.4f}, {xyz_sum[1]:.4f}, {xyz_sum[2]:.4f}]\n"
        log_entry += f"     Magnitude ratio (t2/t1): {np.linalg.norm(xyz_2)/np.linalg.norm(xyz_1):.4f}\n\n"

        log_entry += f"Temporal Analysis:\n"
        log_entry += f"  Queue size: {queue_size} {'(NEW CHUNK)' if is_new_chunk else ''}\n"
        if continuity_gap is not None:
            log_entry += f"  Continuity gap (prev_t2 → curr_t1): {continuity_gap:.4f}\n"

        log_entry += f"Current State:\n"
        log_entry += f"  Current EE pos: [{current_ee_pos[0]:.4f}, {current_ee_pos[1]:.4f}, {current_ee_pos[2]:.4f}]\n"
        log_entry += f"  Current EE 6D:  {rotation_matrix_to_6d(current_ee_rot)}\n"

        if cube_pos is not None:
            log_entry += f"  Cube position:  [{cube_pos[0]:.4f}, {cube_pos[1]:.4f}, {cube_pos[2]:.4f}]\n"

            # Check if action points toward cube
            # xyz_1 is a DELTA (relative position change), not absolute position
            action_direction = xyz_1  # Already a direction vector (delta)
            cube_direction = cube_pos - current_ee_pos  # Direction from EE to cube

            # Normalize directions
            action_dir_norm = action_direction / (np.linalg.norm(action_direction) + 1e-8)
            cube_dir_norm = cube_direction / (np.linalg.norm(cube_direction) + 1e-8)

            # Dot product indicates alignment
            alignment = np.dot(action_dir_norm, cube_dir_norm)

            log_entry += f"\nDirection Analysis (xyz_1 is DELTA, not absolute):\n"
            log_entry += f"  Action delta:     [{action_direction[0]:.4f}, {action_direction[1]:.4f}, {action_direction[2]:.4f}]\n"
            log_entry += f"  Cube direction:   [{cube_direction[0]:.4f}, {cube_direction[1]:.4f}, {cube_direction[2]:.4f}]\n"
            log_entry += f"  Alignment (dot):  {alignment:.4f} {'✓ Points toward cube' if alignment > 0.5 else '✗ Not aligned'}\n"
            log_entry += f"  Distance to cube: {np.linalg.norm(cube_direction):.4f}m\n"
            log_entry += f"  Target EE (curr + delta): [{(current_ee_pos[0]+xyz_1[0]):.4f}, {(current_ee_pos[1]+xyz_1[1]):.4f}, {(current_ee_pos[2]+xyz_1[2]):.4f}]\n"
    else:
        log_entry += f"⚠️  Action vector too short: {len(actions_np)}, expected 20 for EE6D\n"

    log_file.write(log_entry)
    log_file.flush()

    # Print summary to console
    if step % 10 == 0 or args.verbose:
        print(f"\n--- Step {step} ---")
        print(f"  Raw actions (first 7): {actions_np[:7]}")
        if len(actions_np) >= 20:
            print(f"  Target EE pos: [{xyz_1[0]:.3f}, {xyz_1[1]:.3f}, {xyz_1[2]:.3f}]")
            print(f"  Current EE pos: [{current_ee_pos[0]:.3f}, {current_ee_pos[1]:.3f}, {current_ee_pos[2]:.3f}]")
            if cube_pos is not None:
                print(f"  Cube pos: [{cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f}]")
                print(f"  Alignment: {alignment:.3f}")

    # Manual control: user controls joints via UI sliders (data.ctrl is set by viewer)
    # VLA actions are logged but not applied (proper IK/impedance control not yet implemented)

    # 4. Step simulation
    mujoco.mj_step(model, data)

    # 5. Viewer sync and trajectory visualization
    if hasattr(policy, '_queues') and 'action' in policy._queues:
        action_queue = policy._queues['action']
        if len(action_queue) > 0:
            # Extract XYZ deltas from queued actions (up to 10 markers)
            trajectory_deltas = []
            for queued_action in list(action_queue)[:10]:
                # Each action is 20D, extract first XYZ (indices 0-2)
                if isinstance(queued_action, torch.Tensor):
                    delta_xyz = queued_action.flatten()[:3].cpu().numpy()
                else:
                    delta_xyz = np.array(queued_action).flatten()[:3]
                trajectory_deltas.append(delta_xyz)

            # Update mocap body positions to show trajectory markers
            cumulative_pos = current_ee_pos.copy()
            for i, delta_xyz in enumerate(trajectory_deltas):
                cumulative_pos = cumulative_pos + delta_xyz
                try:
                    mocap_id = model.body(f'traj_marker_{i}').mocapid[0]
                    if mocap_id >= 0:
                        data.mocap_pos[mocap_id] = cumulative_pos
                except:
                    pass

            # Hide unused markers (move them underground)
            for i in range(len(trajectory_deltas), 10):
                try:
                    mocap_id = model.body(f'traj_marker_{i}').mocapid[0]
                    if mocap_id >= 0:
                        data.mocap_pos[mocap_id] = [0, 0, -10]
                except:
                    pass

            # Print trajectory preview every 10 steps
            if step % 10 == 0 or is_new_chunk:
                print(f"\n  🎯 Predicted Trajectory (next {len(trajectory_deltas)} steps):")
                print(f"     Current EE: [{current_ee_pos[0]:.3f}, {current_ee_pos[1]:.3f}, {current_ee_pos[2]:.3f}]")
                if cube_pos is not None:
                    print(f"     Cube pos:   [{cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f}]")

                cumulative_pos = current_ee_pos.copy()
                for i, delta_xyz in enumerate(trajectory_deltas[:5]):  # Show first 5
                    if i == 0:
                        print(f"     Delta[{i}]: [{delta_xyz[0]:.4f}, {delta_xyz[1]:.4f}, {delta_xyz[2]:.4f}]")

                    cumulative_pos = cumulative_pos + delta_xyz
                    dist_to_cube = np.linalg.norm(cumulative_pos - cube_pos) if cube_pos is not None else 0
                    marker = "🟢" if i < 2 else "🟡" if i < 4 else "🔴"
                    print(f"     {marker} +{i+1}: [{cumulative_pos[0]:.3f}, {cumulative_pos[1]:.3f}, {cumulative_pos[2]:.3f}] "
                          f"(dist to cube: {dist_to_cube:.3f}m)")
                if len(trajectory_deltas) > 5:
                    print(f"     ... ({len(trajectory_deltas) - 5} more steps in queue)")

    viewer.sync()

    if not viewer.is_running():
        print(f"\nViewer closed at step {step}")
        break

    step += 1

print("\n" + "=" * 60)
print("Demo complete!")
print("=" * 60)
print(f"\n📝 Detailed action log saved to: xvla_action_debug.log")

# Cleanup
log_file.close()
viewer.close()

print("\n🔍 Debugging Summary:")
print("  ✓ Verified X-VLA action mode configuration")
print("  ✓ Logged raw 20D action vectors")
print("  ✓ Decoded EE positions, rotations, and gripper")
print("  ✓ Tracked cube position and alignment")
print("  ⚠️  Still need to implement proper EE→Joint conversion (IK)")
