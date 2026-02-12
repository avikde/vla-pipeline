#!/usr/bin/env python3
"""
SmolVLA Inference Demo with PyBullet

Demonstrates language-conditioned robot control using SmolVLA with PyBullet simulation.
Uses PyBullet's tinyrenderer for fast synthetic camera rendering.

NOTE: This script uses the SO-101 robot and scene setup
      - Robot: SO-101 URDF from https://github.com/TheRobotStudio/SO-ARM100
      - Scene: Cubes manually created at positions from so101_vision_scene.xml
      - Cameras: Positioned to match the MuJoCo camera viewpoints
"""

import argparse
import numpy as np
import torch
from PIL import Image
import time
import pybullet as p
import pybullet_data
import signal

# Parse command line arguments
parser = argparse.ArgumentParser(description='SmolVLA Inference Demo with PyBullet')
parser.add_argument('-l', '--live', action='store_true', help='Enable live visualization (GUI mode)')
parser.add_argument('--steps', type=int, default=100, help='Number of simulation steps (default: 100)')
parser.add_argument('--single-camera', action='store_true', help='Use only one camera view (faster, may reduce accuracy)')
args = parser.parse_args()

# Signal handler for clean exit
def signal_handler(sig, frame):
    print('\n\nInterrupted by user. Cleaning up...')
    try:
        p.disconnect()
    except:
        pass

signal.signal(signal.SIGINT, signal_handler)

print("Loading SmolVLA policy...")
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

# Initialize PyBullet
print("Initializing PyBullet...")
if args.live:
    # GUI mode for live visualization
    physics_client = p.connect(p.GUI)
    print("Live visualization enabled (PyBullet GUI)")
else:
    # Headless mode with DIRECT connection (faster)
    physics_client = p.connect(p.DIRECT)
    print("Running in headless mode (DIRECT)")

# Set up simulation
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
p.setTimeStep(1./240.)  # 240 Hz simulation
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)

# Load plane (floor)
plane_id = p.loadURDF("plane.urdf")

# Load SO-101 robot from URDF
# URDF sourced from: https://github.com/TheRobotStudio/SO-ARM100
print("Loading SO-101 robot from URDF...")
robot_urdf_path = "assets/so101/so101.urdf"
robot_id = p.loadURDF(robot_urdf_path, [0, 0, 0], useFixedBase=True)

# Get number of joints
num_joints = p.getNumJoints(robot_id)
print(f"Robot loaded with {num_joints} joints")

# Find controllable joints (revolute/prismatic)
controllable_joints = []
for i in range(num_joints):
    joint_info = p.getJointInfo(robot_id, i)
    joint_type = joint_info[2]
    if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
        controllable_joints.append(i)

num_dof = min(len(controllable_joints), 6)  # Use first 6 DOF for VLA
print(f"Using {num_dof} controllable joints: {controllable_joints[:num_dof]}")

# Create cubes manually to match SO-101 MuJoCo scene
# Scene positions from assets/so101/so101_vision_scene.xml:
# - Red cube: pos="0.25 0 0.03" size="0.02 0.02 0.02" (0.02m = 2cm cube)
# - Blue cube: pos="0.15 0.15 0.03" size="0.02 0.02 0.02"

# Create red cube
red_cube_size = 0.02  # 2cm cube (half-extents)
red_cube_pos = [0.25, 0.0, 0.03]
red_cube_mass = 0.05  # 50g, matching MuJoCo scene

# Create collision shape and visual shape
red_collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[red_cube_size]*3)
red_visual_shape = p.createVisualShape(
    p.GEOM_BOX,
    halfExtents=[red_cube_size]*3,
    rgbaColor=[1, 0, 0, 1]
)
red_cube_id = p.createMultiBody(
    baseMass=red_cube_mass,
    baseCollisionShapeIndex=red_collision_shape,
    baseVisualShapeIndex=red_visual_shape,
    basePosition=red_cube_pos
)

# Create blue cube
blue_cube_size = 0.02  # 2cm cube
blue_cube_pos = [0.15, 0.15, 0.03]
blue_cube_mass = 0.05  # 50g

blue_collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[blue_cube_size]*3)
blue_visual_shape = p.createVisualShape(
    p.GEOM_BOX,
    halfExtents=[blue_cube_size]*3,
    rgbaColor=[0, 0, 1, 1]
)
blue_cube_id = p.createMultiBody(
    baseMass=blue_cube_mass,
    baseCollisionShapeIndex=blue_collision_shape,
    baseVisualShapeIndex=blue_visual_shape,
    basePosition=blue_cube_pos
)

# Camera rendering setup (256x256 for VLA)
WIDTH, HEIGHT = 256, 256

# Define camera viewpoints matching SO-101 MuJoCo scene
# Camera positions from assets/so101/so101_vision_scene.xml:
# - third_person: pos="0.6 0.4 0.4"
# - top_down: pos="0.2 0 0.8"
# - wrist_cam: pos="0.2 0 0.3"
camera_configs = {
    'third_person': {
        'target': [0.2, 0.0, 0.2],  # Look at workspace center
        'distance': 0.6,
        'yaw': 35,
        'pitch': -25,
    },
    'top_down': {
        'target': [0.2, 0.0, 0.0],  # Look down at workspace
        'distance': 0.8,
        'yaw': 0,
        'pitch': -89,  # Almost straight down
    },
    'wrist_cam': {
        'target': [0.25, 0.0, 0.1],  # Near red cube
        'distance': 0.25,
        'yaw': 90,
        'pitch': -35,
    }
}

def render_camera(camera_name):
    """
    Render from a specific camera using PyBullet's tinyrenderer.
    Returns RGB image as numpy array.
    """
    config = camera_configs[camera_name]

    # Compute view matrix
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=config['target'],
        distance=config['distance'],
        yaw=config['yaw'],
        pitch=config['pitch'],
        roll=0,
        upAxisIndex=2
    )

    # Compute projection matrix
    fov = 60
    aspect = WIDTH / HEIGHT
    near = 0.1
    far = 5.0
    projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

    # Render with tinyrenderer (ER_TINY_RENDERER is fastest)
    width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
        WIDTH, HEIGHT,
        view_matrix,
        projection_matrix,
        renderer=p.ER_TINY_RENDERER  # Use tinyrenderer
    )

    # Convert to numpy array (remove alpha channel)
    rgb_array = np.array(rgb_img, dtype=np.uint8).reshape(HEIGHT, WIDTH, 4)
    rgb_array = rgb_array[:, :, :3]  # Remove alpha channel

    return rgb_array

def preprocess_image(rgb_image, target_size=256, device='cpu'):
    """
    Preprocess image for VLA input.
    - Convert to tensor
    - Normalize to [0, 1]
    - Move to specified device

    Note: Assumes input is already at target_size (256x256)
    """
    # Convert directly to tensor [3, 256, 256]
    if rgb_image.shape[0] == target_size and rgb_image.shape[1] == target_size:
        img_tensor = torch.from_numpy(rgb_image).permute(2, 0, 1).float() / 255.0
    else:
        # Need to resize
        pil_img = Image.fromarray(rgb_image)
        pil_img = pil_img.resize((target_size, target_size), Image.Resampling.BILINEAR)
        img_tensor = torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).float() / 255.0

    # Add batch dimension [1, 3, 256, 256]
    img_tensor = img_tensor.unsqueeze(0)

    # Move to device
    img_tensor = img_tensor.to(device)

    return img_tensor

# Load SmolVLA policy (pretrained on SO-101)
print("Loading SmolVLA pretrained model (~1.8GB)...")
print("This may take a minute on first run...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base").to(device).eval()
print("✓ SmolVLA loaded successfully")

# Create preprocessor and postprocessor
print("Creating SmolVLA preprocessor...")
preprocessor, postprocessor = make_smolvla_pre_post_processors(
    policy.config,
    dataset_stats=None
)

# Language task instruction
task_instruction = "Pick the red cube and place it on the blue cube"
print(f"\nTask: '{task_instruction}'")

# Settle physics
print("\nSettling physics...")
for _ in range(240):  # ~1 second at 240 Hz
    p.stepSimulation()

# Get initial joint positions
initial_joint_states = []
for joint_idx in controllable_joints[:num_dof]:
    joint_state = p.getJointState(robot_id, joint_idx)
    initial_joint_states.append(joint_state[0])

# Simulation loop
print("Running VLA inference loop...")
if args.live:
    print("Close the PyBullet GUI window to stop")
num_steps = args.steps
action_history = []

# Profiling data
profile_data = {
    'render': [],
    'preprocess': [],
    'vla_inference': [],
    'physics': [],
    'total': []
}

for step in range(num_steps):
    iter_start = time.time()

    # 1. Render camera(s)
    render_start = time.time()
    if args.single_camera:
        # Use only third-person view (3x faster rendering)
        img_third = render_camera('third_person')
        render_time = time.time() - render_start
    else:
        # Render all 3 cameras (better accuracy, trained configuration)
        img_third = render_camera('third_person')
        img_top = render_camera('top_down')
        img_wrist = render_camera('wrist_cam')
        render_time = time.time() - render_start

    # 2. Preprocess images for VLA (move to device)
    preprocess_start = time.time()
    img_third_tensor = preprocess_image(img_third, device=device)

    if args.single_camera:
        # Reuse same camera view for all 3 inputs (SmolVLA expects 3 views)
        img_top_tensor = img_third_tensor
        img_wrist_tensor = img_third_tensor
    else:
        img_top_tensor = preprocess_image(img_top, device=device)
        img_wrist_tensor = preprocess_image(img_wrist, device=device)

    # Get current joint positions
    current_joint_positions = []
    for joint_idx in controllable_joints[:num_dof]:
        joint_state = p.getJointState(robot_id, joint_idx)
        current_joint_positions.append(joint_state[0])

    # Pad to 6 DOF if necessary
    while len(current_joint_positions) < 6:
        current_joint_positions.append(0.0)

    # Create observation dict (use camera1, camera2, camera3 for SmolVLA)
    observation = {
        'observation.images.camera1': img_third_tensor,  # Third person
        'observation.images.camera2': img_top_tensor,     # Top down
        'observation.images.camera3': img_wrist_tensor,   # Wrist cam
        'observation.state': torch.from_numpy(np.array(current_joint_positions[:6])).float().unsqueeze(0).to(device),
        'task': task_instruction,
    }

    # Preprocess observation
    processed_obs = preprocessor(observation)
    preprocess_time = time.time() - preprocess_start

    # 3. VLA inference
    vla_start = time.time()
    with torch.inference_mode():
        actions = policy.select_action(processed_obs)

    # Sync GPU if using CUDA
    if device == "cuda":
        torch.cuda.synchronize()
    vla_time = time.time() - vla_start

    # Convert actions to numpy and clip
    if isinstance(actions, torch.Tensor):
        actions_np = actions.detach().cpu().numpy().flatten()
    else:
        actions_np = np.array(actions).flatten()

    # Extract 6 DOF (VLA outputs 20D, we use first 6)
    robot_actions = actions_np[:6]

    # Store for visualization
    action_history.append(robot_actions.copy())

    # Apply actions to robot (position control with PyBullet)
    # Scale down actions for stability
    physics_start = time.time()
    for i, joint_idx in enumerate(controllable_joints[:num_dof]):
        if i < len(robot_actions):
            target_position = current_joint_positions[i] + robot_actions[i] * 0.1
            target_position = np.clip(target_position, -np.pi, np.pi)
            p.setJointMotorControl2(
                robot_id,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=target_position,
                force=50
            )

    # Step simulation (multiple substeps for stability)
    for _ in range(4):  # 4 substeps per control step
        p.stepSimulation()

    physics_time = time.time() - physics_start

    # Total iteration time
    total_time = time.time() - iter_start

    # Store profiling data
    profile_data['render'].append(render_time)
    profile_data['preprocess'].append(preprocess_time)
    profile_data['vla_inference'].append(vla_time)
    profile_data['physics'].append(physics_time)
    profile_data['total'].append(total_time)

    if step % 20 == 0:
        print(f"  Step {step}/{num_steps}: actions = [{robot_actions[0]:.3f}, {robot_actions[1]:.3f}, {robot_actions[2]:.3f}] ({total_time*1000:.1f} ms)")

    # Small delay for GUI mode
    if args.live:
        time.sleep(0.01)

# Print timing statistics
num_completed = len(profile_data['total'])
print(f"\n✓ Completed {num_completed} simulation steps")

if num_completed > 0:
    # Calculate statistics for each component
    print("\n" + "=" * 60)
    print("Performance Breakdown (average per iteration)")
    print("=" * 60)

    render_label = 'Rendering (1 camera)' if args.single_camera else 'Rendering (3 cameras)'
    components = [
        (render_label, 'render'),
        ('Image preprocessing', 'preprocess'),
        ('VLA inference', 'vla_inference'),
        ('Physics step', 'physics'),
        ('Total iteration', 'total')
    ]

    total_avg = np.mean(profile_data['total']) * 1000

    for label, key in components:
        times = profile_data[key]
        avg = np.mean(times) * 1000
        min_t = np.min(times) * 1000
        max_t = np.max(times) * 1000
        percentage = (avg / total_avg * 100) if total_avg > 0 else 0

        print(f"  {label:.<30} {avg:>7.2f} ms  ({percentage:>5.1f}%)")
        if key == 'total':
            print(f"    {'(min/max)':.<28} {min_t:>7.2f} / {max_t:.2f} ms")

    print("\n  Effective rate: {:.1f} Hz".format(1000/total_avg if total_avg > 0 else 0))

# Print final state
red_cube_pos, red_cube_orn = p.getBasePositionAndOrientation(red_cube_id)
print(f"\nFinal red cube position: {red_cube_pos}")

final_joint_positions = []
for joint_idx in controllable_joints[:num_dof]:
    joint_state = p.getJointState(robot_id, joint_idx)
    final_joint_positions.append(joint_state[0])
print(f"Final robot joint positions: {final_joint_positions}")

print("\n" + "="*60)
print("Demo complete!")
print("="*60)

# Disconnect PyBullet and exit cleanly
print("Disconnecting PyBullet...")
try:
    p.disconnect()
except:
    pass

