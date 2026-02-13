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
from PIL import Image
import time
import signal
import sys

# Parse command line arguments
parser = argparse.ArgumentParser(description='SmolVLA Inference Demo with SO-101 Robot')
parser.add_argument('--steps', type=int, default=100, help='Number of simulation steps (default: 100)')
parser.add_argument('--single-camera', action='store_true', help='Use only one camera view (faster, may reduce accuracy)')
parser.add_argument('--headless', action='store_true', help='Run without GUI (no mujoco viewer)')
args = parser.parse_args()

# Import MuJoCo viewer if GUI mode
viewer = None
if not args.headless:
    try:
        import mujoco.viewer as mj_viewer
        HAS_VIEWER = True
    except Exception as e:
        print(f"Warning: mujoco.viewer not available: {e}")
        print("  Falling back to headless mode")
        HAS_VIEWER = False
        args.headless = True
else:
    HAS_VIEWER = False

# Signal handler for clean exit
def signal_handler(sig, frame):
    print('\n\nInterrupted. Cleaning up...')
    if viewer is not None:
        viewer.close()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

print("Loading SmolVLA policy...")
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

# Load SO-101 robot model with vision scene
print("Loading SO-101 MuJoCo model...")
model = mujoco.MjModel.from_xml_path('assets/so101/so101_vision_scene.xml')
data = mujoco.MjData(model)

# Two renderers: 512x512 for the main (third_person) view, 256x256 for the rest
MAIN_WIDTH, MAIN_HEIGHT = 512, 512
VLA_WIDTH, VLA_HEIGHT = 256, 256

# Ensure offscreen framebuffer is large enough for 512x512 rendering
model.vis.global_.offwidth = max(model.vis.global_.offwidth, MAIN_WIDTH)
model.vis.global_.offheight = max(model.vis.global_.offheight, MAIN_HEIGHT)

renderer_main = mujoco.Renderer(model, height=MAIN_HEIGHT, width=MAIN_WIDTH)
renderer_vla = mujoco.Renderer(model, height=VLA_HEIGHT, width=VLA_WIDTH)

def render_camera(camera_name):
    """Render from a specific camera. third_person renders at 512x512 and
    downsamples, others render at 256x256 directly.

    Returns (rgb_array_256x256, render_time).
    """
    camera_id = model.camera(camera_name).id

    if camera_name == 'third_person':
        t0 = time.time()
        renderer_main.update_scene(data, camera=camera_id)
        pixels = renderer_main.render()
        render_time = time.time() - t0

        pil_img = Image.fromarray(pixels)
        pil_img = pil_img.resize((VLA_WIDTH, VLA_HEIGHT), Image.Resampling.BILINEAR)
        return np.array(pil_img), render_time
    else:
        t0 = time.time()
        renderer_vla.update_scene(data, camera=camera_id)
        pixels = renderer_vla.render()
        render_time = time.time() - t0
        return pixels, render_time

def preprocess_image(rgb_image, target_size=256, device='cpu'):
    """
    Preprocess image for VLA input.
    - Convert to tensor
    - Normalize to [0, 1]
    - Move to specified device
    """
    if rgb_image.shape[0] == target_size and rgb_image.shape[1] == target_size:
        img_tensor = torch.from_numpy(rgb_image).permute(2, 0, 1).float() / 255.0
    else:
        pil_img = Image.fromarray(rgb_image)
        pil_img = pil_img.resize((target_size, target_size), Image.Resampling.BILINEAR)
        img_tensor = torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).float() / 255.0

    img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.to(device)
    return img_tensor

# Load SmolVLA policy (pretrained on SO-101)
print("Loading SmolVLA pretrained model (~1.8GB)...")
print("This may take a minute on first run...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base").to(device).eval()
print("✓ SmolVLA loaded successfully")
print(f"  Action chunk size: {policy.config.chunk_size}, n_action_steps: {policy.config.n_action_steps}")
print(f"  Denoising steps: {policy.config.num_steps}")

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

# Launch viewer if GUI mode
if not args.headless and HAS_VIEWER:
    viewer = mj_viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)
    viewer.cam.distance = 0.6
    viewer.cam.azimuth = 35
    viewer.cam.elevation = -25
    viewer.cam.lookat[:] = [0.2, 0.0, 0.2]
    print("MuJoCo viewer launched")

# Simulation loop
print("Running VLA inference loop...")
if viewer is not None:
    print("Close the MuJoCo viewer window to stop early")
else:
    print("Press Ctrl-C to stop early")
num_steps = args.steps
action_history = []

# Profiling data
profile_data = {
    'render': [],
    'render_calls': [],
    'preprocess': [],
    'vla_inference': [],
    'physics': [],
    'viewer': [],
    'total': []
}

for step in range(num_steps):
    iter_start = time.time()

    # 1. Render camera(s)
    render_start = time.time()
    render_call_time = 0.0
    if args.single_camera:
        img_third, t = render_camera('third_person')
        render_call_time += t
        render_time = time.time() - render_start
    else:
        img_third, t = render_camera('third_person')
        render_call_time += t
        img_top, t = render_camera('top_down')
        render_call_time += t
        img_wrist, t = render_camera('wrist_cam')
        render_call_time += t
        render_time = time.time() - render_start

    # 2. Preprocess images for VLA (move to device)
    preprocess_start = time.time()
    img_third_tensor = preprocess_image(img_third, device=device)

    if args.single_camera:
        img_top_tensor = img_third_tensor
        img_wrist_tensor = img_third_tensor
    else:
        img_top_tensor = preprocess_image(img_top, device=device)
        img_wrist_tensor = preprocess_image(img_wrist, device=device)

    observation = {
        'observation.images.camera1': img_third_tensor,
        'observation.images.camera2': img_top_tensor,
        'observation.images.camera3': img_wrist_tensor,
        'observation.state': torch.from_numpy(data.qpos[:6]).float().unsqueeze(0).to(device),
        'task': task_instruction,
    }

    processed_obs = preprocessor(observation)
    preprocess_time = time.time() - preprocess_start

    # 3. VLA inference
    vla_start = time.time()
    queue_was_empty = len(policy._queues.get("action", [])) == 0
    if (queue_was_empty):
        print("  !! New chunk")
    with torch.inference_mode():
        actions = policy.select_action(processed_obs)

    if device == "cuda":
        torch.cuda.synchronize()
    vla_time = time.time() - vla_start

    # Convert actions to numpy
    if isinstance(actions, torch.Tensor):
        actions_np = actions.detach().cpu().numpy().flatten()
    else:
        actions_np = np.array(actions).flatten()

    robot_actions = actions_np[:6]
    action_history.append(robot_actions.copy())

    # Apply actions to robot (position control)
    data.ctrl[:6] = np.clip(robot_actions * 0.1, -1.0, 1.0)

    # 4. Step simulation (physics)
    physics_start = time.time()
    mujoco.mj_step(model, data)
    physics_time = time.time() - physics_start

    # 5. Viewer sync
    viewer_start = time.time()
    if viewer is not None:
        viewer.sync()
        if not viewer.is_running():
            print(f"\nViewer closed at step {step}")
            break
    viewer_time = time.time() - viewer_start

    # Total iteration time
    total_time = time.time() - iter_start

    # Store profiling data
    profile_data['render'].append(render_time)
    profile_data['render_calls'].append(render_call_time)
    profile_data['preprocess'].append(preprocess_time)
    profile_data['vla_inference'].append(vla_time)
    profile_data['physics'].append(physics_time)
    profile_data['viewer'].append(viewer_time)
    profile_data['total'].append(total_time)

    if step % 20 == 0:
        print(f"  Step {step}/{num_steps}: actions = [{robot_actions[0]:.3f}, {robot_actions[1]:.3f}, {robot_actions[2]:.3f}] ({total_time*1000:.1f} ms)")

    if total_time > 0.5:
        print(f"  !! SLOW step {step} ({total_time*1000:.1f} ms){queue_was_empty} — "
              f"render={render_time*1000:.1f}ms, preprocess={preprocess_time*1000:.1f}ms, "
              f"vla={vla_time*1000:.1f}ms, physics={physics_time*1000:.1f}ms, viewer={viewer_time*1000:.1f}ms")

# Close viewer
if viewer is not None:
    viewer.close()

# Print timing statistics
num_completed = len(profile_data['total'])
print(f"\n✓ Completed {num_completed} simulation steps")

if num_completed > 0:
    print("\n" + "=" * 60)
    print("Performance Breakdown (average per iteration)")
    print("=" * 60)

    num_cams = 1 if args.single_camera else 3
    render_label = f'Rendering ({num_cams} camera{"" if num_cams == 1 else "s"})'
    components = [
        (render_label, 'render'),
        ('  render() calls', 'render_calls'),
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

# Print final state
red_cube_id = model.body('red_cube').id
cube_pos = data.xpos[red_cube_id]
print(f"\nFinal cube position: {cube_pos}")
print(f"Final robot joint positions: {data.qpos[:6]}")

print("\n" + "="*60)
print("Demo complete!")
print("="*60)
sys.exit(0)
