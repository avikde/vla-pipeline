
- FIXMEs in IK in widowx_control.py
- Can we write the code in anything else?
  - webgpu?
  - webasm? wasm? - mujoco written in this
- the targets look like thick cylinders
- free camera orientation looks wrong (table is sideways)
- widowx arm looks wrong - render one segment at a time to get right?
- The Gemini model name is set to `gemini-2.0-flash` (general availability) rather than `gemini-robotics-er-1.5-preview` (which may require allowlisting)
- Test if Gemini ER can recognize obstacles and give us info about it

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
