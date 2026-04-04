# Embodied reasoning hierarchical robotics pipeline demo

End-to-end Vision-Language-Action (VLA) models bundle perception, reasoning, and motor control into a single network, but that means the camera, kinematics, and training scenarios are all baked in together. This could cause [unexpected](https://www.avikde.me/debugging-as-architecture-insight) and [unresolvable](https://www.avikde.me/a-coding-agent-equivalent-for-robotics) issues when the task, embodiment, or environment change.

This demo combines the flexible task programming and reasoning of Gemini ER (what is the scene, and what should I do?) and classical camera calibration, kinematics, motion controllers. Gemini blocks are blue, and classical blocks are green. Each layer is independently swappable, and the AI model doesn't need to know anything about the robot's embodiment. This recreates the modularity of a [Sense-Plan-Act](https://www.avikde.me/the-architecture-behind-end-to-end) architecture while retaining the semantic reasoning of a foundation AI model.

```mermaid
flowchart LR
    subgraph SENSE("SENSE")
        P("👁️ Perception\n(Gemini)")
    end
    subgraph PLAN("PLAN")
        TR("🧠 Task\nReasoning\n(Gemini)")
        SU("📐 Spatial\nUnderstanding\n(camera geometry)")
        PA("⚙️ Planning\n& Avoidance\n(kinematics)")
    end
    subgraph ACT("ACT")
        M("🤖 Motors")
    end

    P --> TR
    TR --> SU
    SU --> PA
    PA --> M

    style P fill:#2563eb,stroke:#1d4ed8,color:#fff
    style TR fill:#2563eb,stroke:#1d4ed8,color:#fff
    style SU fill:#16a34a,stroke:#15803d,color:#fff
    style PA fill:#16a34a,stroke:#15803d,color:#fff
    style M fill:#16a34a,stroke:#15803d,color:#fff
```

## Web demo (start here)

[![Live Demo](https://img.shields.io/badge/Live_Demo-Try_it_now-2ea44f?style=for-the-badge)](https://avikde.github.io/vla-pipeline/)

[![Demo screenshot](web/snapshot.png)](https://avikde.github.io/vla-pipeline/)

Try the browser-based demo with MuJoCo WASM + Three.js, no installation required:
- Grab your own [Gemini API key](https://ai.google.dev/gemini-api/docs/api-key) (free tier), or use the pre-baked fallback plan
- Click "Run Task" or "Use Cached Task" and watch!
- Use the mouse to orbit the camera, and check the console for debug logs

### Generalization capabilities: why this is unique

- **Task:** Gemini ER can be prompted with any task and can break down multi-step tasks like "put away the blocks where they belong"
- **Arm embodiment:** Since we use explicit forward kinematics and Jacobians for control, the method does not need any retraining for different hardware
- **Camera position:** Since we use explicit camera geometry to transform Gemini's perception results from image space to 3D space, a different camera can be resolved by calibrating intrinsics and extrinsics using well-understood methods.

### Some tasks to try

> Put the blocks on matching coasters

Should reason that blocks go on color-matched plates

> Swap the green and red blocks

Multi-step plan to move one out of the way first

> Stack the blocks

Move multiple blocks to the same position. Note that since the controller layer assumes each release is at the same tabletop height, the release can be clumsy after the first block.

### Limitations

- Gemini ER's planning capabilities are designed for a top-down view. Therefore, we have to assume a nominal grasp and release height, which is reasonable for tabletop manipulation, except for the block stacking task.
- The interface between Gemini ER and the lower-level planner just conveys a grasping location as a point. This could be augmented with a dedicated grasp generation network initialized with the object center.

## Develop locally (optional)

The `web/` directory contains a fully client-side embodied reasoning demo using MuJoCo WASM + Three.js. No backend required.

```sh
git clone https://github.com/avikde/vla-pipeline.git
cd vla-pipeline
```

`brew install node`

```bash
node web/serve.js
# Open http://localhost:8080
```

## Architecture

### Browser demo (`web/`)

| Module | Role |
|--------|------|
| `web/main.js` | Entry point: init, Gemini pipeline, waypoint sequencing, animation loop |
| `web/mujoco-scene.js` | MuJoCo WASM init, Three.js rendering, MjvScene sync |
| `web/ik-solver.js` | WidowX IK solver |
| `web/gemini-er.js` | Gemini ER scene understanding and task planning |
| `web/prebaked-plan.js` | Fallback plan and detections recorded from a successful Gemini ER run |
| `web/math-utils.js` | Linear algebra, rotation math, pixel-to-3D projection |

Stack: [`@mujoco/mujoco`](https://www.npmjs.com/package/@mujoco/mujoco) WASM (CDN), [Three.js](https://threejs.org/) v0.170 (CDN), Gemini API via `fetch()`.

### WidowX arm

**Action representation (EE6D):** 10D per timestep = [x, y, z, r1x, r1y, r1z, r2x, r2y, r2z, gripper]. The 6D rotation uses two columns of the rotation matrix (third reconstructed via cross product).

### Controller

End-effector motion uses a vector field controller with obstacle avoidance. A repulsive potential field based on a 1/r² relationship pushes the end-effector away from obstacles, while an attractive field pulls it toward the goal waypoint. The combined gradient gives a smooth velocity command that is mapped to joint velocities via the pseudoinverse Jacobian, naturally steering around obstacles without explicit path planning.

## Acknowledgements

- **WidowX model:** From [google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/trossen_wx250s)
- **Google Gemini Robotics ER** [model and description](https://ai.google.dev/gemini-api/docs/robotics-overview)
- **Claude Code** was used for implementation and debugging
