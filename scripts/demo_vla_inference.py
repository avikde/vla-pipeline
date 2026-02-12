#!/usr/bin/env python3
"""
SmolVLA Inference Demo with SO-101 Robot

Demonstrates language-conditioned robot control using SmolVLA.
The pretrained SmolVLA model is already trained on SO-101 data and works out-of-the-box.
"""

import argparse
import mujoco
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import time

# Parse command line arguments
parser = argparse.ArgumentParser(description='SmolVLA Inference Demo with SO-101 Robot')
parser.add_argument('-l', '--live', action='store_true', help='Enable live visualization (requires display)')
parser.add_argument('--steps', type=int, default=100, help='Number of simulation steps (default: 100)')
parser.add_argument('--single-camera', action='store_true', help='Use only one camera view (faster, may reduce accuracy)')
args = parser.parse_args()

# Import MuJoCo viewer if live visualization is enabled
if args.live:
    try:
        import mujoco.viewer as mj_viewer
        HAS_VIEWER = True
        print("Live visualization enabled (using MuJoCo passive viewer)")
    except Exception as e:
        print(f"Warning: MuJoCo viewer not available: {e}")
        print("  Live visualization disabled")
        HAS_VIEWER = False
        args.live = False
else:
    HAS_VIEWER = False

print("Loading SmolVLA policy...")
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

# Load SO-101 robot model with vision scene
print("Loading SO-101 MuJoCo model...")
model = mujoco.MjModel.from_xml_path('assets/so101/so101_vision_scene.xml')
data = mujoco.MjData(model)

# Create renderer at VLA input resolution (256x256) to avoid unnecessary downscaling
WIDTH, HEIGHT = 256, 256
renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)

def render_camera(camera_name):
    """Render from a specific camera and return RGB image."""
    camera_id = model.camera(camera_name).id
    renderer.update_scene(data, camera=camera_id)
    pixels = renderer.render()
    return pixels

def preprocess_image(rgb_image, target_size=256, device='cpu'):
    """
    Preprocess image for VLA input.
    - Convert to tensor
    - Normalize to [0, 1]
    - Move to specified device

    Note: Assumes input is already at target_size (256x256)
    """
    # Convert directly to tensor [3, 256, 256] - skip PIL conversion if already correct size
    if rgb_image.shape[0] == target_size and rgb_image.shape[1] == target_size:
        # Already correct size, convert directly to tensor
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
# Use GPU (PyTorch 2.10.0 + CUDA 12.8 has full Blackwell support!)
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
for _ in range(100):
    mujoco.mj_step(model, data)

# Simulation loop
print("Running VLA inference loop...")
if args.live:
    print("Close the MuJoCo viewer window to stop early")
num_steps = args.steps
action_history = []

# Profiling data
profile_data = {
    'render': [],
    'preprocess': [],
    'vla_inference': [],
    'physics': [],
    'viewer': [],
    'total': []
}

# Launch MuJoCo passive viewer if live visualization is enabled
viewer = None
if args.live and HAS_VIEWER:
    viewer = mj_viewer.launch_passive(model, data)
    viewer.cam.distance = 2.0  # Adjust camera distance
    viewer.cam.azimuth = 45    # Adjust camera angle
    viewer.cam.elevation = -20

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

    # Create observation dict (use camera1, camera2, camera3 for SmolVLA)
    observation = {
        'observation.images.camera1': img_third_tensor,  # Third person
        'observation.images.camera2': img_top_tensor,     # Top down (or duplicate if single-camera)
        'observation.images.camera3': img_wrist_tensor,   # Wrist cam (or duplicate if single-camera)
        'observation.state': torch.from_numpy(data.qpos[:6]).float().unsqueeze(0).to(device),  # 6 DOF
        'task': task_instruction,  # String, not list
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

    # Extract 6 DOF for SO-101 (VLA outputs 20D, we use first 6)
    robot_actions = actions_np[:6]

    # Store for visualization
    action_history.append(robot_actions.copy())

    # Apply actions to robot (position control)
    # Scale down actions for stability
    data.ctrl[:6] = np.clip(robot_actions * 0.1, -1.0, 1.0)

    # 4. Step simulation (physics)
    physics_start = time.time()
    mujoco.mj_step(model, data)
    physics_time = time.time() - physics_start

    # 5. Live visualization - sync viewer with simulation
    viewer_start = time.time()
    if args.live and HAS_VIEWER and viewer is not None:
        viewer.sync()

        # Check if viewer was closed
        if not viewer.is_running():
            print(f"\nViewer closed at step {step}")
            break
    viewer_time = time.time() - viewer_start

    # Total iteration time
    total_time = time.time() - iter_start

    # Store profiling data
    profile_data['render'].append(render_time)
    profile_data['preprocess'].append(preprocess_time)
    profile_data['vla_inference'].append(vla_time)
    profile_data['physics'].append(physics_time)
    profile_data['viewer'].append(viewer_time)
    profile_data['total'].append(total_time)

    if step % 20 == 0:
        print(f"  Step {step}/{num_steps}: actions = [{robot_actions[0]:.3f}, {robot_actions[1]:.3f}, {robot_actions[2]:.3f}] ({total_time*1000:.1f} ms)")

# Close MuJoCo viewer if it was used
if viewer is not None:
    viewer.close()

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
        ('Viewer sync', 'viewer'),
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

# Visualization
print("\nGenerating visualization...")
fig = plt.figure(figsize=(16, 10))

# Camera views
ax1 = plt.subplot(2, 3, 1)
ax1.imshow(render_camera('third_person'))
ax1.set_title('Third Person View - SO-101', fontsize=12)
ax1.axis('off')

ax2 = plt.subplot(2, 3, 2)
ax2.imshow(render_camera('top_down'))
ax2.set_title('Top Down View', fontsize=12)
ax2.axis('off')

ax3 = plt.subplot(2, 3, 3)
ax3.imshow(render_camera('wrist_cam'))
ax3.set_title('Wrist Camera View', fontsize=12)
ax3.axis('off')

# Action history plots
action_history_np = np.array(action_history)

ax4 = plt.subplot(2, 3, 4)
for i in range(min(3, action_history_np.shape[1])):
    ax4.plot(action_history_np[:, i], label=f'Joint {i+1}', linewidth=2)
ax4.set_xlabel('Simulation Step', fontsize=10)
ax4.set_ylabel('Action Value', fontsize=10)
ax4.set_title('VLA-Predicted Actions (Joints 1-3)', fontsize=12)
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

ax5 = plt.subplot(2, 3, 5)
for i in range(3, min(6, action_history_np.shape[1])):
    ax5.plot(action_history_np[:, i], label=f'Joint {i+1}', linewidth=2)
ax5.set_xlabel('Simulation Step', fontsize=10)
ax5.set_ylabel('Action Value', fontsize=10)
ax5.set_title('VLA-Predicted Actions (Joints 4-6)', fontsize=12)
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)

# Info text
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')
cameras_used = "1 camera (third person only)" if args.single_camera else "3 cameras (all views)"
info_text = f"""SmolVLA Inference Demo

Model: lerobot/smolvla_base
Parameters: 450M
Device: {device.upper()}
Task: {task_instruction}

Robot: SO-101 (6 DOF)
Simulation Steps: {num_steps}

Camera Setup: {cameras_used}
• Resolution: {WIDTH}x{HEIGHT} (VLA native)

VLA processes all 3 camera views
and predicts 6D actions for the
SO-101 robot arm.

SmolVLA is pretrained on SO-101
data - no fine-tuning needed!"""
ax6.text(0.1, 0.5, info_text, fontsize=10, verticalalignment='center',
         family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('vla_inference_demo.png', dpi=150, bbox_inches='tight')
print("✓ Saved visualization to vla_inference_demo.png")

# Print final state
red_cube_id = model.body('red_cube').id
cube_pos = data.xpos[red_cube_id]
print(f"\nFinal cube position: {cube_pos}")
print(f"Final robot joint positions: {data.qpos[:6]}")

print("\n" + "="*60)
print("Demo complete! Check vla_inference_demo.png for results.")
print("="*60)
