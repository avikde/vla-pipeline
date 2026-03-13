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
Jacobian damped-least-squares, run on a scratch MjData (does not modify the
live sim state). Optionally solves for full 6-DOF pose or position-only.

PUBLIC API
----------
  solve_ik(model, data, action_trajectory) -> ctrl[0:7] or None
  apply_control(data, ctrl_target)
"""

import numpy as np
import mujoco

# Ordered arm joint names matching ctrl[0..5]
ARM_JOINTS = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
GRIPPER_CTRL_IDX = 6

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
# IK
# ---------------------------------------------------------------------------

def solve_ik(
    model,
    data,
    action_trajectory: list[np.ndarray],
    action_idx: int = 0,
    max_iter: int = 120,
    tol: float = 1e-4,
    damping: float = 1e-4,
    use_orientation: bool = True,
) -> np.ndarray | None:
    """Solve IK for one action from the trajectory and return ctrl[0:7].

    Runs on a scratch MjData copy — the live sim `data` is never modified.

    Args:
        model:             MuJoCo model.
        data:              Live sim data (read-only; used for initial qpos seed).
        action_trajectory: List of 10D ee6d arrays [xyz, rot6d, gripper].
        action_idx:        Which action in the trajectory to target (default 0 = next).
        max_iter:          Max IK iterations.
        tol:               Position convergence threshold [m].
        damping:           Damped-least-squares regularization.
        use_orientation:   Whether to also match target rotation (6-DOF IK).

    Returns:
        ctrl_target: float32 array of shape (7,) — arm joint positions + gripper ctrl,
                     or None if the trajectory is empty.
    """
    if not action_trajectory:
        return None

    action = action_trajectory[action_idx]
    target_pos, target_rot = ee6d_to_pos_rot(action)
    gripper_val = float(action[9])

    # Work on a scratch copy so we don't disturb the live sim
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    scratch.qvel[:] = 0.0
    mujoco.mj_forward(model, scratch)

    ee_id       = model.body(EE_BODY).id
    jnt_ids     = [model.joint(n).id for n in ARM_JOINTS]
    qpos_addrs  = [model.jnt_qposadr[j] for j in jnt_ids]
    dof_addrs   = [model.jnt_dofadr[j]  for j in jnt_ids]
    n_arm       = len(ARM_JOINTS)

    for _ in range(max_iter):
        mujoco.mj_forward(model, scratch)

        pos_err = target_pos - scratch.xpos[ee_id]
        if np.linalg.norm(pos_err) < tol:
            break

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, scratch, jacp, jacr, ee_id)

        Jp = jacp[:, dof_addrs]   # (3, 6)

        if use_orientation:
            Jr = jacr[:, dof_addrs]   # (3, 6)
            R_curr = scratch.xmat[ee_id].reshape(3, 3)
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
        dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(m), err)

        for i, addr in enumerate(qpos_addrs):
            scratch.qpos[addr] += dq[i]

        # Clamp to joint limits
        for i, jid in enumerate(jnt_ids):
            lo, hi = model.jnt_range[jid]
            scratch.qpos[qpos_addrs[i]] = np.clip(scratch.qpos[qpos_addrs[i]], lo, hi)

    joint_positions = np.array([scratch.qpos[a] for a in qpos_addrs], dtype=np.float32)
    ctrl_target = np.zeros(7, dtype=np.float32)
    ctrl_target[0:6] = joint_positions
    ctrl_target[6]   = gripper_action_to_ctrl(gripper_val)
    return ctrl_target


# ---------------------------------------------------------------------------
# Applying control
# ---------------------------------------------------------------------------

def apply_control(data, ctrl_target: np.ndarray) -> None:
    """Write ctrl_target (7,) into data.ctrl for the position servos."""
    data.ctrl[:7] = ctrl_target
