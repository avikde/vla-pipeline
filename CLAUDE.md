# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@README.md

## Blog posts for context

- https://www.avikde.me/p/the-architecture-behind-end-to-end
- https://www.avikde.me/p/debugging-as-architecture-insight

No test suite or linter configured. Pyright is set up for type checking (standard mode).

## Architecture

**Inference loop** (`demo_xvla_widowx_debug.py`):
1. Render `primary` camera at 4:3 (342×256), squish to 256×256 to match BridgeData distortion; second image is black (BridgeData's cam2 is often black)
2. Pack images + 8D proprioceptive state (XYZ, RPY, pad, gripper) + language tokens into observation
3. Run X-VLA → 20D action vector (2 timesteps × 10D)
4. Decode each 10D slice: XYZ target + 6D rotation + gripper value
5. In dry-run mode, visualize the EE trajectory
6. In non-dry-run mode, close the control loop:
  a. Feed EE target to Jacobian IK (`widowx_control.py`) → joint angles → MuJoCo actuators
  b. To be implemented

**Key modules:**
- `scripts/xvla_policy.py` — policy loading, observation building, action decoding, MuJoCo utilities (EE pose, cube position). Also contains `generate_non_vla_reference()` fallback reach-and-grasp policy.
- `scripts/widowx_control.py` — `WidowXController` class: damped least-squares Jacobian IK, 6-DOF or position-only mode, gripper mapping, joint limit enforcement.
- `assets/widowx/` — MuJoCo XML models; `widowx_vision_scene.xml` is the primary scene (wooden table, `primary` camera, `third_person` debug camera).

**Action representation (EE6D):** 10D per timestep = [x, y, z, r1x, r1y, r1z, r2x, r2y, r2z, gripper]. The 6D rotation uses two columns of the rotation matrix (third reconstructed via cross product).

**Model checkpoint:** `lerobot/xvla-widowx` (HuggingFace, fine-tuned on BridgeData). Downloaded automatically on first run.
