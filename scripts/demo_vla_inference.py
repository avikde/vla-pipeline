#!/usr/bin/env python3
"""
SmolVLA Inference Demo with SO-101 Robot

Demonstrates language-conditioned robot control using SmolVLA.
The pretrained SmolVLA model is already trained on SO-101 data and works out-of-the-box.
"""

import mujoco
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

print("Loading SmolVLA policy...")
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

# Load SO-101 robot model with vision scene
print("Loading SO-101 MuJoCo model...")
model = mujoco.MjModel.from_xml_path('assets/so101/so101_vision_scene.xml')
data = mujoco.MjData(model)

# Create renderer
WIDTH, HEIGHT = 640, 480
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
    - Convert to PIL Image
    - Resize to target_size x target_size (256x256 for SmolVLA)
    - Convert to tensor
    - Normalize to [0, 1]
    - Move to specified device
    """
    # Convert numpy to PIL
    pil_img = Image.fromarray(rgb_image)

    # Resize to 256x256 (SmolVLA requirement)
    pil_img = pil_img.resize((target_size, target_size), Image.Resampling.BILINEAR)

    # Convert to tensor [3, 256, 256]
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
num_steps = 100
action_history = []

for step in range(num_steps):
    # Render from 3 cameras
    img_third = render_camera('third_person')
    img_top = render_camera('top_down')
    img_wrist = render_camera('wrist_cam')

    # Preprocess images for VLA (move to device)
    img_third_tensor = preprocess_image(img_third, device=device)
    img_top_tensor = preprocess_image(img_top, device=device)
    img_wrist_tensor = preprocess_image(img_wrist, device=device)

    # Create observation dict (use camera1, camera2, camera3 for SmolVLA)
    observation = {
        'observation.images.camera1': img_third_tensor,  # Third person
        'observation.images.camera2': img_top_tensor,     # Top down
        'observation.images.camera3': img_wrist_tensor,   # Wrist cam
        'observation.state': torch.from_numpy(data.qpos[:6]).float().unsqueeze(0).to(device),  # 6 DOF
        'task': task_instruction,  # String, not list
    }

    # Preprocess observation
    processed_obs = preprocessor(observation)

    # VLA inference
    with torch.inference_mode():
        actions = policy.select_action(processed_obs)

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

    # Step simulation
    mujoco.mj_step(model, data)

    if step % 20 == 0:
        print(f"  Step {step}/{num_steps}: actions = [{robot_actions[0]:.3f}, {robot_actions[1]:.3f}, {robot_actions[2]:.3f}] ...")

print(f"✓ Completed {num_steps} simulation steps")

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
info_text = f"""SmolVLA Inference Demo

Model: lerobot/smolvla_base
Parameters: 450M
Device: {device.upper()}
Task: {task_instruction}

Robot: SO-101 (6 DOF)
Simulation Steps: {num_steps}

Camera Setup:
• Third person (640x480)
• Top down (640x480)
• Wrist cam (640x480)

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
