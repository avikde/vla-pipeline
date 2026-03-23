# X-VLA IK Integration — Diagnosis & Next Steps

## Status
`--up` reference policy works. X-VLA actions cause the arm to move up instead of toward the block.

## Key findings (2025-03-23)

### Coordinate frame is NOT the main problem
X-VLA outputs XYZ in the same range as MuJoCo EE positions. Sample first-chunk trajectory:

| Point | X | Y | Z |
|---|---|---|---|
| MuJoCo EE (with offset) | 0.165 | 0.000 | 0.143 |
| X-VLA action step 1 | 0.170 | 0.001 | 0.156 |
| X-VLA queue[4] | 0.263 | 0.012 | 0.130 |
| Cube | 0.250 | 0.100 | 0.020 |

The trajectory heads roughly toward the cube (x increasing, z decreasing). No global frame transform needed.

### Root cause 1: `FINGER_TIP_OFFSET` distorts proprio
`xvla_policy.py:173` adds 0.02m along gripper X to the finger midpoint. BridgeData records EE at the finger-link origins (Interbotix SDK `ee_gripper_link` = finger midpoint, no offset). This shifts the reported z down by ~2cm, making the model think the EE is lower than it is. The model's first action corrects upward → arm moves up.

### Root cause 2: `cached_action_targets` never advances
In `demo_xvla_widowx_debug.py`, `cached_action_targets` is populated once per chunk (every 30 steps) from the action queue. The IK always targets `cached_action_targets[0]` (via `action_idx=0` default). The arm gets stuck on the first waypoint for the entire chunk instead of progressing along the trajectory.

Compare with `--up` no-VLA mode where `is_new_chunk` is always True and targets are recomputed from the current EE every step.

### Minor: action normalization
Confirmed IDENTITY normalization — no min/max or mean/std scaling. Gripper channels get sigmoid. XYZ and rot6d pass through raw.

## Next steps

### Fix 1: Remove `FINGER_TIP_OFFSET`
- In `xvla_policy.py`: set `FINGER_TIP_OFFSET = 0.0` (or remove the offset logic in `get_ee_pose`)
- In `widowx_control.py:39`: same constant is duplicated, keep in sync
- This fixes the proprio fed to the model AND the IK target-vs-actual comparison

### Fix 2: Advance IK target through the action chunk
Each sim step, the IK should target the current action (dequeued by `select_action`), not a stale cached first entry. Options:
- **Option A**: Use `actions_np` (the dequeued action) directly as the IK target each step, instead of `cached_action_targets[0]`. Keep `cached_action_targets` only for visualization dots.
- **Option B**: Pop from `cached_action_targets` each step so the IK walks through the trajectory.

Option A is simpler and correct — each step already returns the right action via `select_action`.

### After fixes: evaluate
- Run without `--dry-run` and check if the arm tracks the trajectory toward the block
- Watch for Y-axis tracking — the sample trajectory barely moved in Y (0→0.012) vs cube at Y=0.1. This may indicate the model needs more steps or the reach is a multi-chunk motion.
- If the arm reaches but overshoots/oscillates, tune `--kp` or IK damping.
