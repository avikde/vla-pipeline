/**
 * Main entry point: ties together MuJoCo scene, Gemini ER policy, and IK solver.
 *
 * Port of demo_widowx.py -p gemini-er flow.
 */

import { initScene } from './mujoco-scene.js';
import { WidowXController } from './ik-solver.js';
import { detectAndPlan, planToWaypoints, captureSceneImage } from './gemini-er.js';
import { vec3, sub, norm } from './math-utils.js';

// --- Constants matching demo_widowx.py ---
const GRASP_ROT6D = new Float32Array([0, 0, -1, 0, 1, 0]);
const WAYPOINT_THRESHOLD = 0.02; // 2cm
const WAYPOINT_STALL_LIMIT = 500;
const GRIPPER_WAIT_LIMIT = 100;
const GRIPPER_EMA_ALPHA = 0.15;
const SIM_STEPS_PER_FRAME = 10; // ~0.002s timestep * 10 = 0.02s per frame at 60fps

// --- DOM elements ---
const canvas = document.getElementById('viewer');
const loadingOverlay = document.getElementById('loading-overlay');
const loadingText = document.getElementById('loading-text');
const apiKeyInput = document.getElementById('api-key');
const btnRun = document.getElementById('btn-run');
const btnPrebaked = document.getElementById('btn-prebaked');
const chkFreeCam = document.getElementById('chk-free-cam');
const logDiv = document.getElementById('log');
const waypointListDiv = document.getElementById('waypoint-list');
const statusSim = document.getElementById('status-sim');
const statusStep = document.getElementById('status-step');
const statusEe = document.getElementById('status-ee');

// --- Logging ---
function log(msg, level = 'info') {
  const el = document.createElement('div');
  el.className = level;
  el.textContent = msg;
  logDiv.appendChild(el);
  logDiv.scrollTop = logDiv.scrollHeight;
}

// --- State ---
let scene = null;
let controller = null;
let waypoints = [];
let waypointIdx = 0;
let prevWaypointIdx = -1;
let waypointStallSteps = 0;
let gripperWaitSteps = 0;
let smoothedGripper = 1.0;
let step = 0;
let running = false;
let simReady = false;

// --- Initialization ---
async function init() {
  try {
    scene = await initScene(canvas, (msg) => {
      loadingText.textContent = msg;
    });
    simReady = true;
    loadingOverlay.style.display = 'none';
    statusSim.textContent = 'Sim: ready';
    btnRun.disabled = false;
    log('MuJoCo + Three.js initialized', 'success');

    // Initial render
    scene.updateVisuals();
    scene.render();
  } catch (err) {
    loadingText.textContent = `Error: ${err.message}`;
    log(`Init failed: ${err.message}`, 'error');
    console.error(err);
  }
}

// --- Run Gemini ER + simulation loop ---
async function run(apiKey) {
  if (!simReady || running) return;
  running = true;
  btnRun.disabled = true;
  btnPrebaked.disabled = true;

  try {
    // Create IK controller
    controller = new WidowXController(scene.mj, scene.model, {
      useOrientation: true,
    });
    log('IK controller created', 'success');

    // Capture scene image for Gemini
    log('Capturing scene image...', 'info');
    // Temporarily sync to primary camera for screenshot
    const savedCamState = scene.controls.enabled;
    scene.setFreeCam(false);
    scene.updateVisuals();
    const imageBase64 = captureSceneImage(scene.renderer, scene.scene, scene.camera);
    if (savedCamState) scene.setFreeCam(true);

    // Run Gemini ER detect + plan
    const { detections, planSteps } = await detectAndPlan(apiKey, imageBase64, log);

    // Convert plan to waypoints using camera parameters
    scene.mj.mj_forward(scene.model, scene.data);
    const { camPos, camRot, fovyDeg } = scene.getPrimaryCameraParams();
    waypoints = planToWaypoints(planSteps, camPos, camRot, fovyDeg);
    waypointIdx = 0;
    prevWaypointIdx = -1;
    waypointStallSteps = 0;
    gripperWaitSteps = 0;
    smoothedGripper = 1.0;
    step = 0;

    log(`Generated ${waypoints.length} waypoints`, 'success');
    updateWaypointList();

    // Show waypoint markers
    scene.updateWaypointMarkers(waypoints, waypointIdx);

    // Start simulation loop
    requestAnimationFrame(animationLoop);

  } catch (err) {
    log(`Error: ${err.message}`, 'error');
    console.error(err);
    running = false;
    btnRun.disabled = false;
    btnPrebaked.disabled = false;
  }
}

// --- Waypoint list UI ---
function updateWaypointList() {
  waypointListDiv.innerHTML = '';
  for (let i = 0; i < waypoints.length; i++) {
    const wp = waypoints[i];
    const div = document.createElement('div');
    div.className = 'wp-item' + (i === waypointIdx ? ' active' : '') + (i < waypointIdx ? ' done' : '');
    const g = wp.gripper > 0.5 ? 'open' : 'closed';
    div.textContent = `[${i}] (${wp.xyz[0].toFixed(3)}, ${wp.xyz[1].toFixed(3)}, ${wp.xyz[2].toFixed(3)}) ${g}`;
    waypointListDiv.appendChild(div);
  }
}

// --- Get EE position (finger midpoint, matching Python) ---
function getEePos() {
  const model = scene.model;
  const data = scene.data;
  const lfId = model.body('wx250s/left_finger_link').id;
  const rfId = model.body('wx250s/right_finger_link').id;
  return vec3(
    (data.xpos[lfId * 3] + data.xpos[rfId * 3]) / 2,
    (data.xpos[lfId * 3 + 1] + data.xpos[rfId * 3 + 1]) / 2,
    (data.xpos[lfId * 3 + 2] + data.xpos[rfId * 3 + 2]) / 2,
  );
}

// --- Animation loop ---
function animationLoop() {
  if (!running) return;

  const data = scene.data;

  // Get current EE position
  const currentXyz = getEePos();

  // Waypoint sequencing (matching demo_widowx.py gemini-er path)
  if (waypoints.length > 0 && waypointIdx < waypoints.length) {
    const wp = waypoints[waypointIdx];
    const distToWp = norm(sub(currentXyz, wp.xyz));

    let advance = false;
    if (gripperWaitSteps > 0) {
      gripperWaitSteps--;
    } else if (distToWp < WAYPOINT_THRESHOLD) {
      advance = true;
    } else {
      waypointStallSteps++;
      if (waypointStallSteps >= WAYPOINT_STALL_LIMIT) {
        log(`Stalled at waypoint ${waypointIdx} (dist=${distToWp.toFixed(3)}m), skipping`, 'warn');
        advance = true;
      }
    }

    if (advance && waypointIdx < waypoints.length - 1) {
      const prevGrip = wp.gripper;
      waypointIdx++;
      waypointStallSteps = 0;
      const nextWp = waypoints[waypointIdx];
      if (nextWp.gripper !== prevGrip) {
        gripperWaitSteps = GRIPPER_WAIT_LIMIT;
      }
    }

    if (waypointIdx !== prevWaypointIdx) {
      const wpNow = waypoints[waypointIdx];
      log(`Waypoint ${waypointIdx}/${waypoints.length}: [${wpNow.xyz[0].toFixed(3)}, ${wpNow.xyz[1].toFixed(3)}, ${wpNow.xyz[2].toFixed(3)}] grip=${wpNow.gripper.toFixed(1)}`, 'info');
      prevWaypointIdx = waypointIdx;
      updateWaypointList();
      scene.updateWaypointMarkers(waypoints, waypointIdx);
    }

    // Build 10D action from current waypoint
    const wpNow = waypoints[waypointIdx];
    const action10d = new Float32Array(10);
    action10d[0] = wpNow.xyz[0];
    action10d[1] = wpNow.xyz[1];
    action10d[2] = wpNow.xyz[2];
    action10d.set(GRASP_ROT6D, 3);
    // Smooth gripper
    smoothedGripper += GRIPPER_EMA_ALPHA * (wpNow.gripper - smoothedGripper);
    action10d[9] = smoothedGripper;

    // Solve IK and apply control
    const ctrlTarget = controller.solveIk(data.qpos, action10d);
    if (ctrlTarget) {
      WidowXController.applyControl(data.ctrl, ctrlTarget);
    }

    // Step simulation multiple times per frame
    for (let i = 0; i < SIM_STEPS_PER_FRAME; i++) {
      scene.step();
    }
    step += SIM_STEPS_PER_FRAME;

    // Check if done
    if (waypointIdx >= waypoints.length - 1 && (
      norm(sub(getEePos(), waypoints[waypoints.length - 1].xyz)) < WAYPOINT_THRESHOLD ||
      waypointStallSteps >= WAYPOINT_STALL_LIMIT
    )) {
      log('Demo complete!', 'success');
      running = false;
      btnRun.disabled = false;
      btnPrebaked.disabled = false;
    }
  }

  // Update status bar
  statusStep.textContent = `Step: ${step}`;
  const ee = getEePos();
  statusEe.textContent = `EE: (${ee[0].toFixed(3)}, ${ee[1].toFixed(3)}, ${ee[2].toFixed(3)})`;

  // Render
  scene.updateVisuals();
  scene.render();

  if (running) {
    requestAnimationFrame(animationLoop);
  }
}

// --- Event handlers ---

// Restore API key from localStorage
const savedKey = localStorage.getItem('gemini-api-key');
if (savedKey) apiKeyInput.value = savedKey;

apiKeyInput.addEventListener('input', () => {
  localStorage.setItem('gemini-api-key', apiKeyInput.value);
});

btnRun.addEventListener('click', () => {
  const key = apiKeyInput.value.trim();
  if (!key) {
    log('Please enter a Gemini API key, or use "Use Cached Plan"', 'warn');
    return;
  }
  run(key);
});

btnPrebaked.addEventListener('click', () => {
  run(null); // null key triggers prebaked plan
});

chkFreeCam.addEventListener('change', () => {
  if (scene) scene.setFreeCam(chkFreeCam.checked);
});

// Start initialization
init();
