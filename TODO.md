
- FIXMEs in IK in widowx_control.py
- the targets look like thick cylinders
- free camera orientation looks wrong (table is sideways)
- widowx arm looks wrong - render one segment at a time to get right?
- The Gemini model name is set to `gemini-2.0-flash` (general availability) rather than `gemini-robotics-er-1.5-preview` (which may require allowlisting) — you may want to switch this based on your API key's access

## Non-IK solution

The IK iteratively

- finds a step in EE coords toward the target, `err`
- Uses mujoco to get FK and Jac s.t. `J*dq = v`
- Finds dq in joint coords corresponding `dq = J^{dag} err`
- Steps in the dq direction -> effectively vel control

Vel control solution

- We just don't need to have the IK iterate
- Step in the correct direction
- Could use just J^T, or the pinv
- For pos control robot, have internal q reference
