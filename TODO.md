
- planner to avoid the obstacles
- reorganize the top bar

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
