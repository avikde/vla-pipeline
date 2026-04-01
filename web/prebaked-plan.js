/**
 * Pre-baked plan — fallback when no Gemini API key is provided.
 * Recorded from a successful Gemini ER run on the default scene.
 */

export const PREBAKED_PLAN_TEXT = `The task requires matching blocks with targets of the same color.
1.  **Red Block and Red Target:** Pick up the red block located at [588, 757] and place it on the red target at [103, 521].
2.  **Green Block and (Implicit) Green Target:** There is no green target listed, so the green block is skipped.
3.  **Blue Block and Blue Target:** Pick up the blue block at [626, 549] and place it on the blue target at [311, 437].`;

export const PREBAKED_DETECTIONS = [
  { label: 'red target',    point: [521, 103], box_2d: [435,   0, 650, 201], type: 'target' },
  { label: 'green block',   point: [626, 230], box_2d: [530, 157, 742, 305], type: 'block' },
  { label: 'blue target',   point: [437, 311], box_2d: [340, 212, 532, 407], type: 'target' },
  { label: 'dark cylinder', point: [583, 354], box_2d: [450, 305, 720, 405], type: 'obstacle' },
  { label: 'dark cylinder', point: [448, 459], box_2d: [312, 419, 582, 496], type: 'obstacle' },
  { label: 'blue block',    point: [549, 626], box_2d: [452, 560, 649, 690], type: 'block' },
  { label: 'red block',     point: [757, 588], box_2d: [650, 517, 874, 658], type: 'block' },
];

// Task: "Put the blocks on matching targets"
export const PREBAKED_PLAN = [
  { function: 'move',            args: [588, 757, true] },
  { function: 'move',            args: [588, 757, false] },
  { function: 'setGripperState', args: [false] },
  { function: 'move',            args: [588, 757, true] },
  { function: 'move',            args: [103, 521, true] },
  { function: 'move',            args: [103, 521, false] },
  { function: 'setGripperState', args: [true] },
  { function: 'move',            args: [103, 521, true] },
  { function: 'move',            args: [626, 549, true] },
  { function: 'move',            args: [626, 549, false] },
  { function: 'setGripperState', args: [false] },
  { function: 'move',            args: [626, 549, true] },
  { function: 'move',            args: [311, 437, true] },
  { function: 'move',            args: [311, 437, false] },
  { function: 'setGripperState', args: [true] },
  { function: 'move',            args: [311, 437, true] },
];
