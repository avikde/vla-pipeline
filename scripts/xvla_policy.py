"""
X-VLA policy wrapper: loading, tokenization, preprocessing, inference, decoding.

Also contains:
  - MuJoCo EE state utilities (get_ee_pose, get_ee_state_8d, get_cube_position, …)
  - generate_non_vla_reference() for running the sim loop without the VLA model

Import this module to avoid repeating boilerplate across scripts and notebooks.
"""

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_policy(checkpoint: str = "lerobot/xvla-widowx", device: str | None = None):
    """Load XVLAPolicy and its BART tokenizer. Prints config and validates action_mode.

    Returns (policy, tokenizer, device).
    Raises ImportError if lerobot[xvla] is not installed.
    """
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
    from transformers import AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    policy = XVLAPolicy.from_pretrained(checkpoint).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(policy.config.tokenizer_name)

    cfg = policy.config
    print(f"  ✓ Device: {device}")
    print(f"\n  📋 Policy Configuration:")
    print(f"     - action_mode:         {cfg.action_mode}")
    print(f"     - chunk_size:          {cfg.chunk_size}")
    print(f"     - n_action_steps:      {cfg.n_action_steps}")
    print(f"     - num_denoising_steps: {cfg.num_denoising_steps}")
    print(f"     - use_proprio:         {cfg.use_proprio}")
    print(f"     - max_action_dim:      {cfg.max_action_dim}")

    if cfg.action_mode != "ee6d":
        print(f"\n  ⚠️  WARNING: action_mode is '{cfg.action_mode}', expected 'ee6d'")
        print(f"     X-VLA WidowX is trained with EE actions, not joint actions!")
    else:
        print(f"  ✓ Confirmed: Using EE (end-effector) action mode")

    return policy, tokenizer, device


# ---------------------------------------------------------------------------
# Language
# ---------------------------------------------------------------------------

_token_cache: dict = {}


def tokenize_task(task: str, tokenizer, policy_config, device: str) -> tuple:
    """Tokenize a task string. Results are cached by task string.

    Returns (input_ids, attention_mask) as tensors on device.
    """
    if task not in _token_cache:
        tok = tokenizer(
            task,
            padding="max_length",
            max_length=policy_config.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )
        _token_cache[task] = (
            tok["input_ids"].to(device),
            tok["attention_mask"].to(device),
        )
    return _token_cache[task]


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def preprocess_image(rgb_array: np.ndarray, device: str) -> torch.Tensor:
    """Convert H×W×3 uint8 numpy array to (1, 3, H, W) float tensor on device."""
    tensor = torch.from_numpy(rgb_array).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0).to(device)


# ---------------------------------------------------------------------------
# Observation building
# ---------------------------------------------------------------------------

def build_observation(
    img: np.ndarray,
    img2: np.ndarray,
    ee_state_8d: np.ndarray,
    language_tokens: torch.Tensor,
    language_attention_mask: torch.Tensor,
    device: str,
) -> dict:
    """Pack images + proprio + language into the dict XVLAPolicy expects."""
    return {
        "observation.images.image":  preprocess_image(img, device),
        "observation.images.image2": preprocess_image(img2, device),
        "observation.state": torch.from_numpy(ee_state_8d).float().unsqueeze(0).to(device),
        "observation.language.tokens": language_tokens,
        "observation.language.attention_mask": language_attention_mask,
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def select_action(policy, observation: dict, device: str) -> np.ndarray:
    """Run one policy step. Returns flat float32 numpy action vector."""
    with torch.inference_mode():
        try:
            actions = policy.select_action(observation)
        except Exception as e:
            print(f"[xvla_policy] inference error: {e}")
            actions = torch.zeros(20, device=device)

    if device == "cuda":
        torch.cuda.synchronize()

    if isinstance(actions, torch.Tensor):
        return actions.detach().cpu().numpy().flatten()
    return np.array(actions).flatten()


# ---------------------------------------------------------------------------
# Action decoding (ee6d: 20D = two packed timesteps of 10D)
# ---------------------------------------------------------------------------

def decode_ee6d_action(action_vec: np.ndarray):
    """Decode a 20D EE6D action vector.

    Layout (per timestep, 10D each):
      [0:3]  absolute target XYZ
      [3:9]  absolute target rotation (6D)
      [9]    gripper (sigmoid post-processed, 0=closed 1=open)

    Returns (xyz_1, rot6d_1, gripper_1) for the first timestep,
    or (None, None, None) if the vector is too short.
    """
    if len(action_vec) < 20:
        print(f"[xvla_policy] action vector too short: {len(action_vec)}, expected 20")
        return None, None, None

    xyz_1    = action_vec[0:3]
    rot6d_1  = action_vec[3:9]
    gripper_1 = action_vec[9]
    return xyz_1, rot6d_1, gripper_1


def decode_ee6d_both_timesteps(action_vec: np.ndarray):
    """Return both timesteps as (xyz_1, rot6d_1, grip_1, xyz_2, rot6d_2, grip_2)."""
    if len(action_vec) < 20:
        return (None,) * 6
    return (
        action_vec[0:3],  action_vec[3:9],  action_vec[9],
        action_vec[10:13], action_vec[13:19], action_vec[19],
    )


# ---------------------------------------------------------------------------
# MuJoCo EE state utilities
# (require mujoco to be importable, but mujoco is not imported at module level)
# ---------------------------------------------------------------------------

# MuJoCo gripper_link has X pointing down, Z pointing forward.
# Interbotix SDK (BridgeData) uses Z-forward convention.
# This rotation aligns MuJoCo's frame to Interbotix's ee_gripper_link frame.
_MUJOCO_TO_INTERBOTIX = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=np.float64)


def get_ee_pose(model, data):
    """Return (ee_pos, ee_rot_3x3) for the WidowX end-effector.

    Position is the midpoint between the two finger links (matching
    BridgeData's ee_gripper_link convention).
    Orientation is the gripper_link frame rotated to match Interbotix's
    Z-forward EE convention (used in BridgeData).
    """
    left_id  = model.body("wx250s/left_finger_link").id
    right_id = model.body("wx250s/right_finger_link").id
    ee_pos = (data.xpos[left_id] + data.xpos[right_id]) / 2.0
    ee_rot = data.xmat[model.body("wx250s/gripper_link").id].reshape(3, 3)
    ee_rot = ee_rot @ _MUJOCO_TO_INTERBOTIX
    return ee_pos.copy(), ee_rot.copy()


def rotation_matrix_to_euler(rot_mat: np.ndarray):
    """3×3 rotation matrix → (roll, pitch, yaw) in XYZ convention (BridgeData)."""
    sy = np.sqrt(rot_mat[0, 0] ** 2 + rot_mat[1, 0] ** 2)
    if sy >= 1e-6:
        roll  = np.arctan2(rot_mat[2, 1], rot_mat[2, 2])
        pitch = np.arctan2(-rot_mat[2, 0], sy)
        yaw   = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
    else:
        roll  = np.arctan2(-rot_mat[1, 2], rot_mat[1, 1])
        pitch = np.arctan2(-rot_mat[2, 0], sy)
        yaw   = 0.0
    return roll, pitch, yaw


def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """(roll, pitch, yaw) XYZ Euler → 3×3 rotation matrix."""
    cr, sr = np.cos(roll),  np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy_ = np.cos(yaw),  np.sin(yaw)
    Rx = np.array([[1, 0,  0 ], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0,  1,  0 ], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy_, 0], [sy_, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def rotation_matrix_to_6d(rot_mat: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix → 6D representation (first two columns, flattened)."""
    return rot_mat[:, :2].flatten()


def get_ee_state_8d(model, data) -> np.ndarray:
    """8D EE state matching BridgeData observation.state format.

    [x, y, z, roll, pitch, yaw, 0, gripper]
    """
    ee_pos, ee_rot = get_ee_pose(model, data)
    roll, pitch, yaw = rotation_matrix_to_euler(ee_rot)
    gripper_joint_id = model.joint("left_finger").id
    gripper_pos = data.qpos[gripper_joint_id]
    # Map raw joint position [0.015, 0.037] → [0, 1] to match BridgeData convention
    GRIPPER_CLOSE = 0.015
    GRIPPER_OPEN = 0.037
    gripper_normalized = (gripper_pos - GRIPPER_CLOSE) / (GRIPPER_OPEN - GRIPPER_CLOSE)
    return np.array([
        ee_pos[0], ee_pos[1], ee_pos[2],
        roll, pitch, yaw,
        0.0,
        float(np.clip(gripper_normalized, 0.0, 1.0)),
    ], dtype=np.float32)


def get_cube_position(model, data, cube_name: str = "red_block"):
    """Return the world-frame position of a named body, or None if not found."""
    for name in [cube_name, "red_box", "cube"]:
        try:
            return data.xpos[model.body(name).id].copy()
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Reference policy (no VLA)
# ---------------------------------------------------------------------------

def generate_non_vla_reference(
    state: np.ndarray,
    images: list,
    ground_truth_block_position: np.ndarray,
    step_size: float = 0.02,
    grab_radius: float = 0.05,
) -> np.ndarray:
    """Generate a 20D ee6d action without running the VLA model.

    Uses ground-truth block position to produce a simple reach-and-grasp
    trajectory: move the EE toward the block, then close the gripper.

    Args:
        state:                      8D EE state [x,y,z,roll,pitch,yaw,pad,gripper]
        images:                     list of camera images (unused; kept for API symmetry)
        ground_truth_block_position: (3,) array with block XYZ in world frame
        step_size:                  metres to move per step
        grab_radius:                distance threshold to switch from approach to grasp

    Returns:
        20D numpy float32 array in ee6d format (two identical timesteps).
    """
    current_xyz   = state[0:3].copy()
    current_euler = state[3:6].copy()

    block_pos = np.asarray(ground_truth_block_position, dtype=np.float32)
    direction = block_pos - current_xyz
    dist      = float(np.linalg.norm(direction))

    if dist > grab_radius:
        target_xyz = current_xyz + step_size * direction / (dist + 1e-8)
        gripper    = 1.0   # open — approaching
    else:
        target_xyz = current_xyz  # stay in place
        gripper    = 0.0          # close — grasp

    rot_mat = euler_to_rotation_matrix(*current_euler)
    rot6d   = rotation_matrix_to_6d(rot_mat)

    action = np.zeros(20, dtype=np.float32)
    # Timestep 1
    action[0:3]  = target_xyz
    action[3:9]  = rot6d
    action[9]    = gripper
    # Timestep 2 (same target)
    action[10:13] = target_xyz
    action[13:19] = rot6d
    action[19]    = gripper

    return action
