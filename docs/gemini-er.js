/**
 * Gemini ER policy — port of gemini_er_policy.py.
 *
 * Two-prompt pipeline: detect objects, then generate pick-and-place plan.
 * Uses raw fetch() to call the Gemini API with a user-provided API key.
 */

import {
  pixelToWorld3d, VLA_WIDTH, VLA_HEIGHT, vec3,
} from './math-utils.js';

const HEIGHT_OFFSET = 0.15; // metres above table for "high" moves
const GRASP_HEIGHT = 0.04;  // z for "low" moves (top of block)
const TABLE_Z = 0.02;       // table surface z

const GEMINI_MODEL = 'gemini-robotics-er-1.5-preview';
const API_BASE = 'https://generativelanguage.googleapis.com/v1beta/models';

// --- Pre-baked plan (fallback when no API key provided) ---
// Recorded from a successful Gemini ER run on the default scene.
const PREBAKED_DETECTIONS = [
  { point: [482, 291], label: 'red block' },
  { point: [338, 641], label: 'blue target' },
];

const PREBAKED_PLAN = [
  { function: 'move', args: [291, 482, true] },
  { function: 'setGripperState', args: [true] },
  { function: 'move', args: [291, 482, false] },
  { function: 'setGripperState', args: [false] },
  { function: 'move', args: [291, 482, true] },
  { function: 'move', args: [641, 338, true] },
  { function: 'move', args: [641, 338, false] },
  { function: 'setGripperState', args: [true] },
  { function: 'move', args: [641, 338, true] },
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
 * Render the Three.js scene from the primary camera and return base64 JPEG.
 * @param {THREE.WebGLRenderer} renderer
 * @param {THREE.Scene} scene
 * @param {THREE.Camera} camera - The primary camera (matching MuJoCo's)
 * @returns {string} Base64-encoded JPEG (no data: prefix)
 */
export function captureSceneImage(renderer, scene, camera) {
  // Render to the main canvas
  renderer.render(scene, camera);
  const dataUrl = renderer.domElement.toDataURL('image/jpeg', 0.9);
  // Strip "data:image/jpeg;base64," prefix
  return dataUrl.split(',')[1];
}

// --- Main pipeline ---

/**
 * Run Gemini ER detect + plan pipeline.
 *
 * @param {string|null} apiKey - Gemini API key, or null for prebaked plan
 * @param {string} imageBase64 - Base64 JPEG of scene
 * @param {Function} log - Logging callback (msg, level)
 * @returns {Promise<{detections: Array, planSteps: Array}>}
 */
export async function detectAndPlan(apiKey, imageBase64, log = () => {}) {
  if (!apiKey) {
    log('Using pre-baked plan (no API key)', 'warn');
    return { detections: PREBAKED_DETECTIONS, planSteps: PREBAKED_PLAN };
  }

  // Prompt 1: detect red block + blue target
  const prompt1 = `Locate and point to the red block and the blue target. The label returned
should be an identifying name for the object detected.
The answer should follow the json format: [{"point": <point>,
"label": <label1>}, ...]. The points are in [y, x] format
normalized to 0-1000.`;

  log('Detecting objects [Gemini]...', 'info');
  const resp1 = await geminiCall(apiKey, imageBase64, prompt1);
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
  const resp2 = await geminiCall(apiKey, imageBase64, prompt2);
  log(`Plan response: ${resp2.slice(0, 300)}`, 'info');
  const planSteps = parseJson(resp2);
  log(`Plan has ${planSteps.length} steps`, 'success');

  return { detections, planSteps };
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
