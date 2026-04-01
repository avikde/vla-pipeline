/**
 * Pre-baked plan — fallback when no Gemini API key is provided.
 * Recorded from a successful Gemini ER run on the default scene.
 */

export const PREBAKED_PLAN_TEXT = `The task requires putting away the blocks. The detected objects include three blocks: "green block", "blue block", and "red block". The "red target" and "blue target" likely represent the destinations for the red and blue blocks, respectively. The green block does not have a corresponding target, so I will place it near the other blocks. The "dark cylinder" objects are obstacles to avoid.

I will perform the following steps for each block:
1. Move high above the block to avoid obstacles.
2. Lower the arm and open the gripper.
3. Close the gripper to pick up the block.
4. Move high above the scene again.
5. Move high above the corresponding target location.
6. Lower the arm and open the gripper to release the block.
7. Move high again to prepare for the next action.

Let's process each block:
- **Red block:** Pick up the red block from [588, 755] and place it on the red target at [102, 542].
- **Blue block:** Pick up the blue block from [620, 545] and place it on the blue target at [310, 435].
- **Green block:** Pick up the green block from [230, 638] and place it near the other blocks, for instance, near the initial position of the red block, as there is no specific green target. I will choose a clear area near where the red block was, perhaps around [600, 700].`;

export const PREBAKED_DETECTIONS = [
  { label: 'red target',    point: [542, 102], box_2d: [435,   0, 650, 201], type: 'target' },
  { label: 'blue target',   point: [435, 310], box_2d: [342, 210, 532, 408], type: 'target' },
  { label: 'green block',   point: [638, 230], box_2d: [535, 158, 742, 305], type: 'block' },
  { label: 'dark cylinder', point: [585, 358], box_2d: [450, 305, 722, 405], type: 'obstacle' },
  { label: 'dark cylinder', point: [445, 458], box_2d: [312, 420, 582, 498], type: 'obstacle' },
  { label: 'blue block',    point: [545, 620], box_2d: [452, 560, 650, 690], type: 'block' },
  { label: 'red block',     point: [755, 588], box_2d: [652, 515, 878, 658], type: 'block' },
];

export const PREBAKED_PLAN = [
  { function: 'move',            args: [588, 755, true] },
  { function: 'move',            args: [588, 755, false] },
  { function: 'setGripperState', args: [true] },
  { function: 'setGripperState', args: [false] },
  { function: 'move',            args: [588, 755, true] },
  { function: 'move',            args: [102, 542, true] },
  { function: 'move',            args: [102, 542, false] },
  { function: 'setGripperState', args: [true] },
  { function: 'move',            args: [102, 542, true] },
  { function: 'move',            args: [620, 545, true] },
  { function: 'move',            args: [620, 545, false] },
  { function: 'setGripperState', args: [false] },
  { function: 'move',            args: [620, 545, true] },
  { function: 'move',            args: [310, 435, true] },
  { function: 'move',            args: [310, 435, false] },
  { function: 'setGripperState', args: [true] },
  { function: 'move',            args: [310, 435, true] },
  { function: 'move',            args: [230, 638, true] },
  { function: 'move',            args: [230, 638, false] },
  { function: 'setGripperState', args: [false] },
  { function: 'move',            args: [230, 638, true] },
  { function: 'move',            args: [600, 700, true] },
  { function: 'move',            args: [600, 700, false] },
  { function: 'setGripperState', args: [true] },
  { function: 'move',            args: [600, 700, true] },
];
