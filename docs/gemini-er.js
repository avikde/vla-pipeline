/**
 * Gemini ER policy — port of gemini_er_policy.py.
 *
 * Two-prompt pipeline: detect objects, then generate pick-and-place plan.
 * Uses raw fetch() to call the Gemini API with a user-provided API key.
 */

import {
  pixelToWorld3d, pixelToRay, VLA_WIDTH, VLA_HEIGHT, vec3, sub, norm,
} from './math-utils.js';

const HEIGHT_OFFSET = 0.15; // metres above table for "high" moves
const GRASP_HEIGHT = 0.04;  // z for "low" moves (top of block)
const TABLE_Z = 0.02;       // table surface z

const GEMINI_MODEL = 'gemini-robotics-er-1.5-preview';
const API_BASE = 'https://generativelanguage.googleapis.com/v1beta/models';

/**
 * Convert a Gemini bounding box + center point to a 3D obstacle cylinder.
 *
 * @param {number[]} bbox - [top_y, left_x, bottom_y, right_x] normalized 0-1000
 * @param {number[]} point - [y, x] center detection normalized 0-1000
 * @param {Float64Array} camPos - Camera world position (3)
 * @param {Float64Array} camRot - Camera rotation matrix (9, row-major)
 * @param {number} fovyDeg - Camera FOV Y in degrees
 * @returns {{ center: Float64Array, radius: number, height: number }}
 */
export function bboxToObstacle3d(bbox, point, camPos, camRot, fovyDeg) {
  const [topY, leftX, , rightX] = bbox;
  const [pointNormY, pointNormX] = point;
  const centerX = (leftX + rightX) / 2;

  // Convert normalized 0-1000 → VLA pixel coords
  const toPxX = (v) => v / 1000 * VLA_WIDTH;
  const toPxY = (v) => v / 1000 * VLA_HEIGHT;

  // Base position from center detection point — more reliable than bbox bottom edge,
  // which projects to the front face of the object rather than its center footprint.
  const pBase = pixelToWorld3d(toPxX(pointNormX), toPxY(pointNormY), camPos, camRot, fovyDeg, TABLE_Z);

  // Footprint radius: project left/right bbox edges at the detection point's y level
  const pLeft = pixelToWorld3d(toPxX(leftX), toPxY(pointNormY), camPos, camRot, fovyDeg, TABLE_Z);
  const pRight = pixelToWorld3d(toPxX(rightX), toPxY(pointNormY), camPos, camRot, fovyDeg, TABLE_Z);
  const radius = Math.max(norm(sub(pRight, pLeft)) / 2, 0.01);

  // Height: cast ray through top-center, find z where it's directly above pBase.
  // Use the dominant ray component for numerical stability.
  const topRay = pixelToRay(toPxX(centerX), toPxY(topY), camPos, camRot, fovyDeg);
  const t = Math.abs(topRay.dir[0]) >= Math.abs(topRay.dir[1])
    ? (pBase[0] - topRay.origin[0]) / topRay.dir[0]
    : (pBase[1] - topRay.origin[1]) / topRay.dir[1];
  const topZ = topRay.origin[2] + t * topRay.dir[2];
  const height = Math.max(topZ - TABLE_Z, 0.01);

  return {
    center: vec3(pBase[0], pBase[1], TABLE_Z + height / 2),
    radius,
    height,
  };
}

// --- Pre-baked plan (fallback when no API key provided) ---
// Recorded from a successful Gemini ER run on the default scene.
const PREBAKED_DETECTIONS = [
  { point: [755, 585], label: 'red block' },
  { point: [461, 311], label: 'blue target' },
];

const PREBAKED_PLAN = [
  { function: 'move', args: [585, 755, true] },
  { function: 'setGripperState', args: [true] },
  { function: 'move', args: [585, 755, false] },
  { function: 'setGripperState', args: [false] },
  { function: 'move', args: [585, 755, true] },
  { function: 'move', args: [311, 461, true] },
  { function: 'move', args: [311, 461, false] },
  { function: 'setGripperState', args: [true] },
  { function: 'move', args: [311, 461, true] },
];

// --- Gemini API call ---

/**
 * Call Gemini with image + text prompt. Returns raw text response.
 *
 * @param {string} apiKey - User's Gemini API key
 * @param {string} imageBase64 - Base64-encoded JPEG image
 * @param {string} prompt - Text prompt
 * @returns {Promise<string>} Raw text response
 */
async function geminiCall(apiKey, imageBase64, prompt) {
  const url = `${API_BASE}/${GEMINI_MODEL}:generateContent?key=${apiKey}`;
  const body = {
    contents: [{
      parts: [
        {
          inline_data: {
            mime_type: 'image/jpeg',
            data: imageBase64,
          },
        },
        { text: prompt },
      ],
    }],
    generationConfig: {
      temperature: 0.5,
      thinkingConfig: { thinkingBudget: 0 },
    },
  };

  const resp = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

  if (!resp.ok) {
    const errText = await resp.text();
    throw new Error(`Gemini API error ${resp.status}: ${errText}`);
  }

  const data = await resp.json();
  return data.candidates?.[0]?.content?.parts?.[0]?.text ?? '';
}

// --- JSON parsing ---

function parseJson(rawText) {
  const raw = rawText.trim();
  // Try markdown JSON fence
  const fenceMatch = raw.match(/```json?\s*([\s\S]*?)\s*```/);
  if (fenceMatch) return JSON.parse(fenceMatch[1]);
  // Fallback: find first JSON array or object
  const arrMatch = raw.match(/[\[{][\s\S]*[\]}]/);
  if (arrMatch) return JSON.parse(arrMatch[0]);
  return JSON.parse(raw);
}

// --- Canvas to base64 JPEG ---

/**
 * Grab the current canvas content as a JPEG for the Gemini prompt.
 * Caller is responsible for rendering the desired camera first
 * (e.g. via mujocoScene.renderPrimaryCamera()).
 * @param {THREE.WebGLRenderer} renderer
 * @returns {string} Base64-encoded JPEG (no data: prefix)
 */
export function captureSceneImage(renderer) {
  const dataUrl = renderer.domElement.toDataURL('image/jpeg', 0.9);
  return dataUrl.split(',')[1];
}

// --- Main pipeline ---

/**
 * Run Gemini ER detect + plan pipeline.
 *
 * @param {string|null} apiKey - Gemini API key, or null for prebaked plan
 * @param {string} imageBase64 - Base64 JPEG of scene
 * @param {Function} log - Logging callback (msg, level)
 * @param {{ camPos: Float64Array, camRot: Float64Array, fovyDeg: number }} [cameraParams]
 *   Camera parameters for 3D obstacle extraction from bounding boxes
 * @returns {Promise<{detections: Array, planSteps: Array, obstacles: Array}>}
 */
export async function detectAndPlan(apiKey, imageBase64, log = () => {}, cameraParams = null) {
  if (!apiKey) {
    log('Using pre-baked plan (no API key)', 'warn');
    return { detections: PREBAKED_DETECTIONS, planSteps: PREBAKED_PLAN, obstacles: [] };
  }

  // Prompt 1: detect all objects with bounding boxes
  const prompt1 = `Detect all objects and obstacles on the table. For each, return:
- "label": a short identifying name (e.g. "red block", "dark cylinder", "blue target mat")
- "point": center location in [y, x] format normalized to 0-1000
- "box_2d": bounding box as [top_y, left_x, bottom_y, right_x] normalized to 0-1000
- "type": one of "block", "target", "obstacle"

Return JSON: [{"label": ..., "point": [y, x], "box_2d": [top_y, left_x, bottom_y, right_x], "type": ...}, ...]`;

  log('Detecting objects [Gemini]...', 'info');
  let t0 = performance.now();
  const resp1 = await geminiCall(apiKey, imageBase64, prompt1);
  log(`Gemini responded in ${((performance.now() - t0) / 1000).toFixed(1)}s`, 'info');
  log(`Detection response: ${resp1.slice(0, 200)}`, 'info');
  const detections = parseJson(resp1);

  const blockDet = detections.find(d => d.label.toLowerCase().includes('block')) ?? detections[0];
  const targetDet = detections.find(d => d.label.toLowerCase().includes('target')) ?? detections[detections.length - 1];
  const [blockY, blockX] = blockDet.point;
  const [targetY, targetX] = targetDet.point;
  log(`Block: (${blockX}, ${blockY})  Target: (${targetX}, ${targetY})`, 'success');

  // Prompt 2: generate pick-and-place plan
  const prompt2 = `You are a robotic arm with six degrees-of-freedom. You have the
following functions available to you:

def move(x, y, high):
  # moves the arm to the given coordinates. The boolean value 'high' set
  to True means the robot arm should be lifted above the scene for
  avoiding obstacles during motion. 'high' set to False means the robot
  arm should have the gripper placed on the surface for interacting with
  objects.

def setGripperState(opened):
  # Opens the gripper if opened set to true, otherwise closes the gripper

Perform a pick and place operation where you pick up the block at
normalized coordinates (${blockX}, ${blockY}) and place it on the
target at normalized coordinates (${targetX}, ${targetY}).
Provide the sequence of function calls as a JSON list of objects, where
each object has a "function" key (the function name) and an "args" key
(a list of arguments for the function).
Also, include your reasoning before the JSON output.
For example:
Reasoning: To pick up the block, I will first move the arm to a high
position above the block, open the gripper, move down to the block,
close the gripper, lift the arm, move to a high position above the bowl,
move down to the bowl, open the gripper, and then lift the arm back to
a high position.`;

  log('Generating plan [Gemini]...', 'info');
  t0 = performance.now();
  const resp2 = await geminiCall(apiKey, imageBase64, prompt2);
  log(`Gemini responded in ${((performance.now() - t0) / 1000).toFixed(1)}s`, 'info');
  log(`Plan response: ${resp2.slice(0, 300)}`, 'info');
  const planSteps = parseJson(resp2);
  log(`Plan has ${planSteps.length} steps`, 'success');

  // Extract 3D obstacle cylinders from bounding boxes
  const obstacles = [];
  if (cameraParams) {
    const { camPos, camRot, fovyDeg } = cameraParams;
    for (const det of detections) {
      if (det.box_2d) {
        const obs3d = bboxToObstacle3d(det.box_2d, det.point, camPos, camRot, fovyDeg);
        obstacles.push({ label: det.label, type: det.type, ...obs3d });
      }
    }
    log(`Extracted ${obstacles.length} 3D obstacles from bounding boxes`, 'info');
    for (const obs of obstacles) {
      log(`  ${obs.label}: center=(${obs.center[0].toFixed(3)}, ${obs.center[1].toFixed(3)}, ${obs.center[2].toFixed(3)}) r=${obs.radius.toFixed(3)} h=${obs.height.toFixed(3)}`, 'info');
    }
  }

  return { detections, planSteps, obstacles };
}

/**
 * Convert plan steps into waypoints: Array<{ xyz: Float64Array(3), gripper: number }>.
 *
 * @param {Array} planSteps - From detectAndPlan
 * @param {Float64Array} camPos - Camera world position (3)
 * @param {Float64Array} camRot - Camera rotation matrix (9, row-major)
 * @param {number} fovyDeg - Camera FOV Y in degrees
 * @returns {Array<{xyz: Float64Array, gripper: number}>}
 */
export function planToWaypoints(planSteps, camPos, camRot, fovyDeg) {
  let gripper = 1.0; // start open
  const waypoints = [];

  for (const step of planSteps) {
    const func = step.function;
    const args = step.args;

    if (func === 'move' && args.length >= 3) {
      const normX = Number(args[0]);
      const normY = Number(args[1]);
      const high = Boolean(args[2]);

      // Convert normalized 0-1000 -> VLA pixel coords
      const pxX = normX / 1000 * VLA_WIDTH;
      const pxY = normY / 1000 * VLA_HEIGHT;

      const worldXy = pixelToWorld3d(pxX, pxY, camPos, camRot, fovyDeg, TABLE_Z);
      worldXy[2] = high ? TABLE_Z + HEIGHT_OFFSET : GRASP_HEIGHT;

      waypoints.push({ xyz: new Float64Array(worldXy), gripper });

    } else if (func === 'setGripperState' && args.length >= 1) {
      const newGripper = args[0] ? 1.0 : 0.0;
      if (newGripper !== gripper) {
        gripper = newGripper;
        if (waypoints.length > 0) {
          const lastXyz = new Float64Array(waypoints[waypoints.length - 1].xyz);
          waypoints.push({ xyz: lastXyz, gripper });
        }
      }
    }
  }

  return waypoints;
}
