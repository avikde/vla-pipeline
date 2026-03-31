
- planner to avoid the obstacles
- reorganize the top bar

## Non-IK solution

The IK iteratively

- finds a step in EE coords toward the target, `err`
- Uses mujoco to get FK and Jac s.t. `J*dq = v`
- Finds dq in joint coords corresponding `dq = J^{dag} err`
- Steps in the dq direction -> effectively vel control

Vel control solution

- We just don't need to have the IK iterate
- Step in the correct direction
- Could use just `J^T`, or the pinv
- For pos control robot, have internal `q` reference

## Planner

Have a local planner to avoid obstacles
- VLA would have to be retrained
- Tie back to adaptation post - LL avoids collisions

Avoidance
- does mujoco have collision avoidance functionality?
- cuboids or ellipses on links?
- VF planner?
- Teach as some FK "obstacle" function

```python
err = ee_pose - target = FK(q) - target
derr = J * dq

# Obst func designed to get large when close to obst
b = obst(q) # scalar
Jo = dobst/dq # gradient, small away from obst

# Add gradients? -> step in
# dq = Jdag * derr + Jo
```
