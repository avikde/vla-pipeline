"""
WidowX WX250S control and utility functions.

Contains:
  - EE state utilities (get_ee_pose, get_ee_state_8d, get_body_position)
  - Rotation math (euler ↔ rotation matrix, 6D representation)
  - Action decoding (ee6d format)
  - Reference trajectory generation (generate_reference_action)
  - IK controller (WidowXController)

ACTUATOR MODEL
--------------
All 7 actuators are MuJoCo <position> servos:
  ctrl[0..5] = target joint positions [rad]  (waist → wrist_rotate)
  ctrl[6]    = target left_finger position [m] (right_finger coupled via equality)
  kp = 50 for arm joints, kp = 200 for gripper
  forcerange = [-35, 35] N·m

There is no torque or velocity control mode in this model.
"""

import numpy as np
import mujoco

# Ordered arm joint names matching ctrl[0..5]
ARM_JOINTS = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]

# left_finger slide range from XML: range="0.015 0.037"
GRIPPER_OPEN  = 0.037  # m
GRIPPER_CLOSE = 0.015  # m

EE_BODY = "wx250s/gripper_link"
LEFT_FINGER_BODY  = "wx250s/left_finger_link"
RIGHT_FINGER_BODY = "wx250s/right_finger_link"
FINGER_TIP_OFFSET = 0.0  # no offset; finger midpoint matches BridgeData ee_gripper_link

# MuJoCo gripper_link has X pointing down, Z pointing forward.
# Interbotix SDK (BridgeData) uses Z-forward convention.
# This rotation aligns MuJoCo's frame to Interbotix's ee_gripper_link frame.
_MUJOCO_TO_INTERBOTIX = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=np.float64)


# ---------------------------------------------------------------------------
# EE state utilities
# ---------------------------------------------------------------------------

def get_ee_pose(model, data):
    """Return (ee_pos, ee_rot_3x3) for the WidowX end-effector.

    Position is the midpoint between the two finger links (matching
    BridgeData's ee_gripper_link convention).
    Orientation is the gripper_link frame rotated to match Interbotix's
    Z-forward EE convention (used in BridgeData).
    """
    left_id  = model.body(LEFT_FINGER_BODY).id
    right_id = model.body(RIGHT_FINGER_BODY).id
    ee_pos = (data.xpos[left_id] + data.xpos[right_id]) / 2.0
    ee_rot = data.xmat[model.body(EE_BODY).id].reshape(3, 3)
    ee_rot = ee_rot @ _MUJOCO_TO_INTERBOTIX
    return ee_pos.copy(), ee_rot.copy()


def get_ee_state_8d(model, data) -> np.ndarray:
    """8D EE state matching BridgeData observation.state format.

    [x, y, z, roll, pitch, yaw, 0, gripper]
    """
    ee_pos, ee_rot = get_ee_pose(model, data)
    roll, pitch, yaw = rotation_matrix_to_euler(ee_rot)
    gripper_joint_id = model.joint("left_finger").id
    gripper_pos = data.qpos[gripper_joint_id]
    gripper_normalized = (gripper_pos - GRIPPER_CLOSE) / (GRIPPER_OPEN - GRIPPER_CLOSE)
    return np.array([
        ee_pos[0], ee_pos[1], ee_pos[2],
        roll, pitch, yaw,
        0.0,
        float(np.clip(gripper_normalized, 0.0, 1.0)),
    ], dtype=np.float32)


def get_body_position(model, data, body_name: str = "red_block") -> np.ndarray | None:
    """Return the world-frame position of a named body, or None if not found."""
    for name in [body_name, "red_box", "cube"]:
        try:
            return data.xpos[model.body(name).id].copy()
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Rotation math
# ---------------------------------------------------------------------------

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
    """3×3 rotation matrix → 6D representation (first two columns, concatenated).

    Layout: [col0_x, col0_y, col0_z, col1_x, col1_y, col1_z]
    This matches the column-major layout expected by ee6d_to_pos_rot.
    """
    return np.concatenate([rot_mat[:, 0], rot_mat[:, 1]])


def interbotix_rot6d_to_mujoco(rot6d: np.ndarray) -> np.ndarray:
    """Convert a 6D rotation from Interbotix (BridgeData) frame to MuJoCo body frame.

    X-VLA outputs rotations in the Interbotix convention (matching its training data).
    The IK solver compares against raw MuJoCo xmat, so we must undo the
    _MUJOCO_TO_INTERBOTIX transform: R_mujoco = R_interbotix @ M^T.
    """
    a1 = rot6d[0:3]
    a2 = rot6d[3:6]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-8)
    b3 = np.cross(b1, b2)
    rot_interbotix = np.column_stack([b1, b2, b3])
    rot_mujoco = rot_interbotix @ _MUJOCO_TO_INTERBOTIX.T
    return rotation_matrix_to_6d(rot_mujoco)


# ---------------------------------------------------------------------------
# Action decoding (ee6d: 20D = two packed timesteps of 10D)
# ---------------------------------------------------------------------------

def ee6d_to_pos_rot(action_10d: np.ndarray):
    """Unpack a 10D ee6d action into (pos [3], rot_3x3 [3,3]).

    Layout: [xyz(3) | rot6d(6) | gripper(1)]
    The 6D rotation is the Gram-Schmidt orthonormalization of two column vectors.
    """
    xyz   = action_10d[0:3]
    a1    = action_10d[3:6]
    a2    = action_10d[6:9]
    b1    = a1 / (np.linalg.norm(a1) + 1e-8)
    b2    = a2 - np.dot(b1, a2) * b1
    b2    = b2 / (np.linalg.norm(b2) + 1e-8)
    b3    = np.cross(b1, b2)
    rot   = np.column_stack([b1, b2, b3])
    return xyz, rot


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
        print(f"[widowx_control] action vector too short: {len(action_vec)}, expected 20")
        return None, None, None

    xyz_1    = action_vec[0:3]
    rot6d_1  = action_vec[3:9]
    gripper_1 = action_vec[9]
    return xyz_1, rot6d_1, gripper_1


def gripper_action_to_ctrl(gripper_val: float) -> float:
    """Map gripper scalar (0 = closed, 1 = open) to left_finger ctrl [m]."""
    return float(np.clip(
        GRIPPER_CLOSE + gripper_val * (GRIPPER_OPEN - GRIPPER_CLOSE),
        GRIPPER_CLOSE, GRIPPER_OPEN,
    ))


# ---------------------------------------------------------------------------
# Reference trajectory generation
# ---------------------------------------------------------------------------

def generate_reference_action(
    state: np.ndarray,
    target_xyz: np.ndarray,
    step_size: float = 0.02,
    grab_radius: float = 0.05,
) -> np.ndarray:
    """Generate a 20D ee6d action for a simple reach-and-grasp policy.

    Moves the EE toward target_xyz, closing the gripper when close enough.
    Orientation is copied from the current state's Euler angles.

    Args:
        state:       8D EE state [x,y,z,roll,pitch,yaw,pad,gripper]
        target_xyz:  (3,) target position in world frame
        step_size:   metres to move per step
        grab_radius: distance threshold to switch from approach to grasp

    Returns:
        20D numpy float32 array in ee6d format (two identical timesteps).
    """
    current_xyz   = state[0:3].copy()
    current_euler = state[3:6].copy()

    block_pos = np.asarray(target_xyz, dtype=np.float32)
    direction = block_pos - current_xyz
    dist      = float(np.linalg.norm(direction))

    if dist > grab_radius:
        next_xyz = current_xyz + step_size * direction / (dist + 1e-8)
        gripper  = 1.0   # open — approaching
    else:
        next_xyz = current_xyz  # stay in place
        gripper  = 0.0          # close — grasp

    rot_mat = euler_to_rotation_matrix(*current_euler)
    rot6d   = rotation_matrix_to_6d(rot_mat)

    action = np.zeros(20, dtype=np.float32)
    # Timestep 1
    action[0:3]  = next_xyz
    action[3:9]  = rot6d
    action[9]    = gripper
    # Timestep 2 (same target)
    action[10:13] = next_xyz
    action[13:19] = rot6d
    action[19]    = gripper

    return action


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class WidowXController:
    """Stateful IK controller for the WidowX WX250S.

    Allocates a single scratch MjData and caches body/joint IDs at construction
    time — both are reused across every solve_ik call.
    """

    def __init__(
        self,
        model,
        max_iter: int = 120,
        tol: float = 1e-4,
        damping: float = 1e-4,
        use_orientation: bool = True,
    ):
        self._model = model
        self._max_iter = max_iter
        self._tol = tol
        self._damping = damping
        self._use_orientation = use_orientation

        # Allocate once
        self._scratch = mujoco.MjData(model)

        # Cache IDs once
        self._ee_id     = model.body(EE_BODY).id
        self._lf_id     = model.body(LEFT_FINGER_BODY).id
        self._rf_id     = model.body(RIGHT_FINGER_BODY).id
        self._jnt_ids   = [model.joint(n).id for n in ARM_JOINTS]
        self._qpos_addrs = [model.jnt_qposadr[j] for j in self._jnt_ids]
        self._dof_addrs  = [model.jnt_dofadr[j]  for j in self._jnt_ids]

        # Home joint positions for IK regularization (prevents branch-switching)
        home_ctrl = model.keyframe('home').ctrl
        self._home_q = np.array([home_ctrl[i] for i in range(6)], dtype=np.float64)

    def _ee_pos(self, d) -> np.ndarray:
        """Compute EE position matching get_ee_pose: finger midpoint + tip offset."""
        mid = (d.xpos[self._lf_id] + d.xpos[self._rf_id]) / 2.0
        ee_rot = d.xmat[self._ee_id].reshape(3, 3)
        return mid + FINGER_TIP_OFFSET * ee_rot[:, 0]

    def solve_ik(
        self,
        qpos: np.ndarray,
        action_trajectory: list[np.ndarray],
        action_idx: int = 0,
        debug: bool = False,
    ) -> np.ndarray | None:
        """Solve IK for one action from the trajectory and return ctrl[0:7].

        Args:
            qpos:              Current generalized positions (copied into scratch first).
            action_trajectory: List of 10D ee6d arrays [xyz, rot6d, gripper].
            action_idx:        Which action in the trajectory to target.
            debug:             If True, print FK validation (target vs achieved EE pos).

        Returns:
            ctrl_target: float32 array of shape (7,) — arm joint positions +
                         gripper ctrl, or None if the trajectory is empty.
        """
        if not action_trajectory:
            return None

        action = action_trajectory[action_idx]
        target_pos, target_rot = ee6d_to_pos_rot(action)
        gripper_val = float(action[9])

        scratch = self._scratch
        scratch.qpos[:] = qpos
        scratch.qvel[:] = 0.0

        iters = 0
        for iters in range(self._max_iter):
            mujoco.mj_forward(self._model, scratch)

            pos_err = target_pos - self._ee_pos(scratch)
            if np.linalg.norm(pos_err) < self._tol:
                break

            jacp = np.zeros((3, self._model.nv))
            jacr = np.zeros((3, self._model.nv))
            mujoco.mj_jacBody(self._model, scratch, jacp, jacr, self._ee_id)

            Jp = jacp[:, self._dof_addrs]   # (3, 6)

            if self._use_orientation:
                Jr = jacr[:, self._dof_addrs]   # (3, 6)
                R_curr = scratch.xmat[self._ee_id].reshape(3, 3)
                R_err  = target_rot @ R_curr.T
                trace  = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
                angle  = np.arccos(trace)
                if angle > 1e-6:
                    s = 1.0 / (2.0 * np.sin(angle))
                    rot_err = angle * s * np.array([
                        R_err[2, 1] - R_err[1, 2],
                        R_err[0, 2] - R_err[2, 0],
                        R_err[1, 0] - R_err[0, 1],
                    ])
                else:
                    rot_err = np.zeros(3)

                J   = np.vstack([Jp, Jr])           # (6, 6)
                # Scale rotation error down so position tracking dominates
                ORI_WEIGHT = 0.2
                err = np.concatenate([pos_err, ORI_WEIGHT * rot_err])
            else:
                J   = Jp                             # (3, 6)
                err = pos_err

            m = J.shape[0]
            Jpinv = J.T @ np.linalg.solve(J @ J.T + self._damping * np.eye(m), np.eye(m))
            dq = Jpinv @ err

            # Bias toward home configuration to prevent branch-switching
            HOME_BIAS = 0.02
            q_curr = np.array([scratch.qpos[a] for a in self._qpos_addrs])
            dq += HOME_BIAS * (self._home_q - q_curr)

            # Clamp joint step to prevent large orientation errors from flinging joints
            MAX_DQ = 0.1  # rad per IK iteration
            dq_norm = np.linalg.norm(dq)
            if dq_norm > MAX_DQ:
                dq *= MAX_DQ / dq_norm

            for i, addr in enumerate(self._qpos_addrs):
                scratch.qpos[addr] += dq[i]

            # Clamp to joint limits
            for i, jid in enumerate(self._jnt_ids):
                lo, hi = self._model.jnt_range[jid]
                scratch.qpos[self._qpos_addrs[i]] = np.clip(scratch.qpos[self._qpos_addrs[i]], lo, hi)

        joint_positions = np.array([scratch.qpos[a] for a in self._qpos_addrs], dtype=np.float32)
        ctrl_target = np.zeros(7, dtype=np.float32)
        ctrl_target[0:6] = joint_positions
        ctrl_target[6]   = gripper_action_to_ctrl(gripper_val)

        if debug:
            # FK validation: run forward kinematics on the solved joint positions
            # (need one extra mj_forward since the last loop iteration may have
            # updated qpos without a corresponding mj_forward)
            mujoco.mj_forward(self._model, scratch)
            achieved_pos = self._ee_pos(scratch).copy()
            residual = np.linalg.norm(target_pos - achieved_pos)
            print(
                f"  [IK] iters={iters+1}/{self._max_iter}  "
                f"target=[{target_pos[0]:.4f}, {target_pos[1]:.4f}, {target_pos[2]:.4f}]  "
                f"achieved=[{achieved_pos[0]:.4f}, {achieved_pos[1]:.4f}, {achieved_pos[2]:.4f}]  "
                f"residual={residual:.6f}m"
            )

        return ctrl_target

    @staticmethod
    def apply_control(ctrl: np.ndarray, ctrl_target: np.ndarray) -> None:
        """Write ctrl_target (7,) into the ctrl array for the position servos."""
        ctrl[:7] = ctrl_target
