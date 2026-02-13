#!/usr/bin/env python3
"""
MuJoCo Rendering & Physics Benchmark

Benchmarks MuJoCo rendering and physics performance without any VLA/torch/lerobot
dependencies. Useful for comparing WSL vs Windows performance, and vs PyBullet.

Dependencies: pip install mujoco numpy pillow
"""

import argparse
import numpy as np
from PIL import Image
import time
import mujoco
import signal
import sys
import platform

# Parse command line arguments
parser = argparse.ArgumentParser(description='MuJoCo Rendering & Physics Benchmark')
parser.add_argument('--steps', type=int, default=100, help='Number of simulation steps (default: 100)')
parser.add_argument('--single-camera', action='store_true', help='Use only one camera view')
parser.add_argument('--headless', action='store_true', help='Run without GUI (no mujoco viewer)')
args = parser.parse_args()

# Import mujoco viewer if GUI mode
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

# Print system info
print("=" * 60)
print("MuJoCo Rendering & Physics Benchmark")
print("=" * 60)
print(f"  Platform: {platform.system()} {platform.release()}")
print(f"  Python:   {platform.python_version()}")
print(f"  MuJoCo:   {mujoco.__version__}")
print(f"  Cameras:  {'1 (single)' if args.single_camera else '3 (all views)'}")
print(f"  GUI:      {'off (headless)' if args.headless else 'on (mujoco viewer)'}")
print(f"  Steps:    {args.steps}")
print("=" * 60)

# Load SO-101 model
print("\nLoading SO-101 MuJoCo model...")
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

# Camera names from the scene XML
CAMERAS = ['third_person', 'top_down', 'wrist_cam']

def render_camera(camera_name):
    """
    Render a camera view. third_person renders at 512x512 and downsamples,
    others render at 256x256 directly.

    Returns (rgb_array_256x256, render_time).
    """
    camera_id = model.camera(camera_name).id

    if camera_name == 'third_person':
        # Render at 512x512 (main view), downsample to 256x256
        t0 = time.time()
        renderer_main.update_scene(data, camera=camera_id)
        pixels = renderer_main.render()
        render_time = time.time() - t0

        # Downsample to VLA input size
        pil_img = Image.fromarray(pixels)
        pil_img = pil_img.resize((VLA_WIDTH, VLA_HEIGHT), Image.Resampling.BILINEAR)
        return np.array(pil_img), render_time
    else:
        # Render directly at 256x256
        t0 = time.time()
        renderer_vla.update_scene(data, camera=camera_id)
        pixels = renderer_vla.render()
        render_time = time.time() - t0

        return pixels, render_time

# Controllable joints (actuators)
num_actuators = model.nu
num_dof = min(num_actuators, 6)
print(f"Robot loaded with {model.njnt} joints, {num_actuators} actuators (using {num_dof})")

# Settle physics
print("Settling physics...")
for _ in range(100):
    mujoco.mj_step(model, data)

# Launch viewer if GUI mode
if not args.headless and HAS_VIEWER:
    viewer = mj_viewer.launch_passive(model, data)
    viewer.cam.distance = 0.6
    viewer.cam.azimuth = 35
    viewer.cam.elevation = -25
    viewer.cam.lookat[:] = [0.2, 0.0, 0.2]
    print("MuJoCo viewer launched")

# Main benchmark loop
print(f"\nRunning {args.steps} steps...")
if viewer is not None:
    print("Close the MuJoCo viewer window to stop early")
else:
    print("Press Ctrl-C to stop early")

profile_data = {
    'render': [],
    'render_calls': [],
    'viewer': [],
    'physics': [],
    'total': []
}

for step in range(args.steps):
    iter_start = time.time()

    # 1. Render cameras
    render_start = time.time()
    render_call_time = 0.0
    if args.single_camera:
        _, t = render_camera('third_person')
        render_call_time += t
    else:
        for cam_name in CAMERAS:
            _, t = render_camera(cam_name)
            render_call_time += t
    render_time = time.time() - render_start

    # 2. Generate random actions (stand-in for VLA)
    robot_actions = np.random.randn(num_dof) * 0.01

    # 3. Apply actions and step physics
    physics_start = time.time()
    data.ctrl[:num_dof] = np.clip(data.qpos[:num_dof] + robot_actions, -1.0, 1.0)
    mujoco.mj_step(model, data)
    physics_time = time.time() - physics_start

    # 4. Viewer sync
    viewer_start = time.time()
    if viewer is not None:
        viewer.sync()
        if not viewer.is_running():
            print(f"\nViewer closed at step {step}")
            break
    viewer_time = time.time() - viewer_start

    total_time = time.time() - iter_start

    profile_data['render'].append(render_time)
    profile_data['render_calls'].append(render_call_time)
    profile_data['viewer'].append(viewer_time)
    profile_data['physics'].append(physics_time)
    profile_data['total'].append(total_time)

    if step % 20 == 0:
        print(f"  Step {step}/{args.steps} ({total_time*1000:.1f} ms)")

# Close viewer
if viewer is not None:
    viewer.close()

# Results
num_completed = len(profile_data['total'])
print(f"\n✓ Completed {num_completed} steps")

if num_completed > 0:
    print("\n" + "=" * 60)
    print("Performance Breakdown (average per iteration)")
    print("=" * 60)

    num_cams = 1 if args.single_camera else 3
    components = [
        (f'Rendering ({num_cams} camera{"" if num_cams == 1 else "s"})', 'render'),
        ('  render() calls', 'render_calls'),
        ('Viewer sync', 'viewer'),
        ('Physics step', 'physics'),
        ('Total iteration', 'total')
    ]

    total_avg = np.mean(profile_data['total']) * 1000

    for label, key in components:
        times = profile_data[key]
        avg = np.mean(times) * 1000
        min_t = np.min(times) * 1000
        max_t = np.max(times) * 1000
        pct = (avg / total_avg * 100) if total_avg > 0 else 0
        print(f"  {label:.<30} {avg:>7.2f} ms  ({pct:>5.1f}%)")
        if key == 'total':
            print(f"    {'(min/max)':.<28} {min_t:>7.2f} / {max_t:.2f} ms")

    print(f"\n  Effective rate: {1000/total_avg:.1f} Hz")

print("\n" + "=" * 60)
print("Done!")
print("=" * 60)
sys.exit(0)
