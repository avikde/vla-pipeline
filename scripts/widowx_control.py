"""
WidowX WX250S control utilities.

ACTUATOR MODEL
--------------
All 7 actuators are MuJoCo <position> servos:
  ctrl[0..5] = target joint positions [rad]  (waist → wrist_rotate)
  ctrl[6]    = target left_finger position [m] (right_finger coupled via equality)
  kp = 50 for arm joints, kp = 200 for gripper
  forcerange = [-35, 35] N·m

There is no torque or velocity control mode in this model.

IK
--
Jacobian damped-least-squares, run on a persistent scratch MjData (does not
modify the live sim state). Optionally solves for full 6-DOF pose or position-only.

PUBLIC API
----------
  WidowXController(model)
    .solve_ik(qpos, action_trajectory) -> ctrl[0:7] or None
    .apply_control(ctrl, ctrl_target)
"""

import numpy as np
import mujoco

# Ordered arm joint names matching ctrl[0..5]
ARM_JOINTS = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]

# left_finger slide range from XML: range="0.015 0.037"
GRIPPER_OPEN  = 0.037  # m
GRIPPER_CLOSE = 0.015  # m

EE_BODY = "wx250s/gripper_link"


# ---------------------------------------------------------------------------
# Action decoding
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


def gripper_action_to_ctrl(gripper_val: float) -> float:
    """Map gripper scalar (0 = closed, 1 = open) to left_finger ctrl [m]."""
    return float(np.clip(
        GRIPPER_CLOSE + gripper_val * (GRIPPER_OPEN - GRIPPER_CLOSE),
        GRIPPER_CLOSE, GRIPPER_OPEN,
    ))


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
        self._jnt_ids   = [model.joint(n).id for n in ARM_JOINTS]
        self._qpos_addrs = [model.jnt_qposadr[j] for j in self._jnt_ids]
        self._dof_addrs  = [model.jnt_dofadr[j]  for j in self._jnt_ids]

    def solve_ik(
        self,
        qpos: np.ndarray,
        action_trajectory: list[np.ndarray],
        action_idx: int = 0,
        debug: bool = False,
    ) -> np.ndarray | None:
        """Solve IK for one action from the trajectory and return ctrl[0:7].

        Args:
            qpos:              Current generalized positions (used as IK seed).
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

            pos_err = target_pos - scratch.xpos[self._ee_id]
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
                err = np.concatenate([pos_err, rot_err])
            else:
                J   = Jp                             # (3, 6)
                err = pos_err

            m = J.shape[0]
            dq = J.T @ np.linalg.solve(J @ J.T + self._damping * np.eye(m), err)

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
            achieved_pos = scratch.xpos[self._ee_id].copy()
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
