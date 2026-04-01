/**
 * Gemini ER policy — port of gemini_er_policy.py.
 *
 * Two-prompt pipeline: detect objects, then generate pick-and-place plan.
 * Uses raw fetch() to call the Gemini API with a user-provided API key.
 */

import {
  pixelToWorld3d, pixelToRay, VLA_WIDTH, VLA_HEIGHT, vec3, sub, norm, dot,
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
 * @param {{ data: Float32Array, width: number, height: number }} [depthBuffer]
 *   Linear view-space depth in metres, row-major, y=0 at top. When provided,
 *   use depth at the detected point instead of assuming TABLE_Z.
 * @returns {{ center: Float64Array, radius: number, height: number }}
 */
export function bboxToObstacle3d(bbox, point, camPos, camRot, fovyDeg, depthBuffer) {
  const [topY, leftX, bottomY, rightX] = bbox;
  const [pointNormY, pointNormX] = point;
  const toPxX = (v) => v / 1000 * VLA_WIDTH;
  const toPxY = (v) => v / 1000 * VLA_HEIGHT;
  // Sample view-space depth at the detected center pixel
  const px = Math.round(pointNormX / 1000 * (depthBuffer.width - 1));
  const py = Math.round(pointNormY / 1000 * (depthBuffer.height - 1));
  const viewDepth = depthBuffer.data[py * depthBuffer.width + px];
  // Camera forward axis in world space: -col2(camRot) (col2 = indices 2,5,8 in row-major)
  const fwd = vec3(-camRot[2], -camRot[5], -camRot[8]);
  const projectAtDepth = (normX, normY, depth) => {
    const ray = pixelToRay(toPxX(normX), toPxY(normY), camPos, camRot, fovyDeg);
    const t = depth / dot(ray.dir, fwd);
    return vec3(camPos[0] + t * ray.dir[0], camPos[1] + t * ray.dir[1], camPos[2] + t * ray.dir[2]);
  };
  const center = projectAtDepth(pointNormX, pointNormY, viewDepth);

  // Project bbox edges at center depth; compute 3D Euclidean span for each axis.
  const centerX = (leftX + rightX) / 2;
  const centerY = (topY + bottomY) / 2;
  const pLeft  = projectAtDepth(leftX,   centerY, viewDepth);
  const pRight = projectAtDepth(rightX,  centerY, viewDepth);
  const pTop   = projectAtDepth(centerX, topY,    viewDepth);
  const pBot   = projectAtDepth(centerX, bottomY, viewDepth);
  const horizontal_size = Math.max(norm(sub(pRight, pLeft)), 0.02);
  const vertical_size   = Math.max(norm(sub(pBot, pTop)),   0.02);
  return { center, horizontal_size, vertical_size };
}

// --- Pre-baked plan (fallback when no API key provided) ---
// Recorded from a successful Gemini ER run on the default scene.
const PREBAKED_DETECTIONS = [
  { label: 'red target mat', point: [520, 100], box_2d: [435,   0, 650, 200], type: 'target' },
  { label: 'blue target mat', point: [430, 305], box_2d: [340, 210, 530, 407], type: 'target' },
  { label: 'green block',    point: [625, 230], box_2d: [535, 160, 740, 305], type: 'block' },
  { label: 'dark cylinder',  point: [585, 360], box_2d: [450, 305, 720, 405], type: 'obstacle' },
  { label: 'dark cylinder',  point: [445, 455], box_2d: [315, 420, 580, 497], type: 'obstacle' },
  { label: 'blue block',     point: [545, 620], box_2d: [455, 560, 650, 690], type: 'block' },
  { label: 'red block',      point: [755, 590], box_2d: [655, 517, 875, 657], type: 'block' },
];

const PREBAKED_PLAN = [
  { function: 'move',            args: [590, 755, true] },
  { function: 'move',            args: [590, 755, false] },
  { function: 'setGripperState', args: [false] },
  { function: 'move',            args: [590, 755, true] },
  { function: 'move',            args: [305, 430, true] },
  { function: 'move',            args: [305, 430, false] },
  { function: 'setGripperState', args: [true] },
  { function: 'move',            args: [305, 430, true] },
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
 * @param {{ camPos: Float64Array, camRot: Float64Array, fovyDeg: number,
 *           depthBuffer: { data: Float32Array, width: number, height: number } }} [cameraParams]
 *   Camera parameters and depth buffer for 3D obstacle extraction
 * @returns {Promise<{detections: Array, planSteps: Array, obstacles: Array}>}
 */
export async function detectAndPlan(apiKey, imageBase64, log = () => {}, cameraParams = null, task) {
  if (!apiKey) {
    log('Using pre-baked plan (no API key)', 'warn');
    const obstacles = [];
    if (cameraParams) {
      const { camPos, camRot, fovyDeg, depthBuffer } = cameraParams;
      for (const det of PREBAKED_DETECTIONS) {
        if (det.box_2d) {
          const obs3d = bboxToObstacle3d(det.box_2d, det.point, camPos, camRot, fovyDeg, depthBuffer);
          obstacles.push({ label: det.label, type: det.type, ...obs3d });
        }
      }
      log(`Extracted ${obstacles.length} 3D obstacles from pre-baked detections`, 'info');
    }
    return { detections: PREBAKED_DETECTIONS, planSteps: PREBAKED_PLAN, obstacles };
  }

  // Prompt 1: detect all objects with bounding boxes
  const prompt1 = `
    Detect all objects on the table. Classify each using these strict rules:
    - "block": a small colored cube or rectangular solid that the robot can pick up and place (e.g. "red block", "green block")
    - "target": a flat colored mat, marker, or region on the table surface indicating a destination (e.g. "blue target", "yellow target mat")
    - "obstacle": any other object that is neither a graspable block nor a target mat (e.g. cylinders, bowls, irregular shapes)

    Do NOT classify a block as an obstacle. Do NOT include the robot arm or gripper.

    For each detected object return:
    - "label": a short name using color + type (e.g. "red block", "blue target", "dark cylinder")
    - "point": center location in [y, x] format normalized to 0-1000
    - "box_2d": bounding box as [top_y, left_x, bottom_y, right_x] normalized to 0-1000
    - "type": one of "block", "target", "obstacle"

    Return JSON: [{"label": ..., "point": [y, x], "box_2d": [top_y, left_x, bottom_y, right_x], "type": ...}, ...]`;

  log('Detecting objects [Gemini]...', 'info');
  let t0 = performance.now();
  const resp1 = await geminiCall(apiKey, imageBase64, prompt1);
  log(`Gemini responded in ${((performance.now() - t0) / 1000).toFixed(1)}s`, 'info');
  log(`Detection response: ${resp1}`, 'info');
  const detections = parseJson(resp1);

  log(`Detections: ${detections.length} objects`, 'success');

  // Prompt 2: task-level plan using full detection list
  // Gemini detection points are [y, x]; move(x, y) takes x first — swap here.
  const detectionsJson = JSON.stringify(
    detections.map(d => ({ label: d.label, type: d.type, point: [d.point[1], d.point[0]] })),
  );
  const prompt2 = `You are a robotic arm with six degrees-of-freedom. You have the
following functions available to you:

def move(x, y, high):
  # moves the arm to the given coordinates (normalized 0-1000).
  # high=True lifts the arm above the scene to avoid obstacles.
  # high=False places the gripper on the surface to interact with objects.

def setGripperState(opened):
  # Opens the gripper if opened=True, otherwise closes it.

Objects detected on the table (normalized 0-1000 pixel coordinates, [x, y] format matching move(x, y)):
${detectionsJson}

Task: ${task}

Use the "point" coordinates from the detected objects above.
Provide the complete sequence of function calls as a JSON list of objects,
where each object has a "function" key and an "args" key (a list, not an object).
Example: [{"function": "move", "args": [586, 760, true]}, ...]
Include brief reasoning before the JSON output.`;

  log('Generating plan [Gemini]...', 'info');
  t0 = performance.now();
  const resp2 = await geminiCall(apiKey, imageBase64, prompt2);
  log(`Gemini responded in ${((performance.now() - t0) / 1000).toFixed(1)}s`, 'info');
  log(`Plan response: ${resp2}`, 'info');
  const planSteps = parseJson(resp2);
  log(`Plan has ${planSteps.length} steps`, 'success');

  // Extract 3D obstacle cylinders from bounding boxes
  const obstacles = [];
  if (cameraParams) {
    const { camPos, camRot, fovyDeg, depthBuffer } = cameraParams;
    for (const det of detections) {
      if (det.box_2d) {
        const obs3d = bboxToObstacle3d(det.box_2d, det.point, camPos, camRot, fovyDeg, depthBuffer);
        obstacles.push({ label: det.label, type: det.type, ...obs3d });
      }
    }
    log(`Extracted ${obstacles.length} 3D obstacles from bounding boxes`, 'info');
    for (const obs of obstacles) {
      log(`  ${obs.label}: center=(${obs.center[0].toFixed(3)}, ${obs.center[1].toFixed(3)}, ${obs.center[2].toFixed(3)})`, 'info');
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
    // Gemini may return args as an array [x,y,high] or object {x,y,high}/{opened}
    const rawArgs = step.args;
    const args = Array.isArray(rawArgs)
      ? rawArgs
      : func === 'move'
        ? [rawArgs.x, rawArgs.y, rawArgs.high]
        : [rawArgs.opened];

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
          waypoints.push({ xyz: lastXyz, gripper, gripperChange: true });
        }
      }
    }
  }

  return waypoints;
}
