/**
 * WidowX + Gemini ER pick-and-place demo.
 *
 * Entry point: initializes MuJoCo WASM + Three.js, wires up UI,
 * runs Gemini ER detection/planning, and animates the arm via IK.
 */

// All module imports are dynamic (inside init) to avoid saturating
// HTTP/1.1 connections during local dev. On GitHub Pages (HTTP/2) this
// wouldn't matter, but it doesn't hurt either.

// --- UI elements ---
const loadingOverlay = document.getElementById('loading-overlay');
const loadingText = document.getElementById('loading-text');
const logDiv = document.getElementById('log');
const btnRun = document.getElementById('btn-run');
const btnPrebaked = document.getElementById('btn-prebaked');
const apiKeyInput = document.getElementById('api-key');
const taskInput = document.getElementById('task-input');
const chkFreeCam = document.getElementById('chk-free-cam');
const statusSim = document.getElementById('status-sim');
const statusStep = document.getElementById('status-step');
const statusEe = document.getElementById('status-ee');
const waypointList = document.getElementById('waypoint-list');

// --- Logging ---
function log(msg, level = '') {
  console.log(msg);
  const el = document.createElement('div');
  el.textContent = msg;
  if (level) el.className = level;
  logDiv.appendChild(el);
  logDiv.scrollTop = logDiv.scrollHeight;
}

// --- State ---
let mujocoScene = null;
let ikController = null;
let waypoints = [];
let currentWpIdx = 0;
let simStep = 0;
let wpSteps = 0; // physics steps spent on current waypoint
let running = false;
let animationId = null;

// IK convergence: steps per waypoint before moving on
const STEPS_PER_WAYPOINT = 200;
const STEPS_GRIPPER_CHANGE = 500; // extra dwell for gripper open/close to settle
const PHYSICS_STEPS_PER_FRAME = 5;

// --- Restore API key from localStorage ---
const savedKey = localStorage.getItem('gemini-api-key');
if (savedKey) apiKeyInput.value = savedKey;
apiKeyInput.addEventListener('input', () => {
  localStorage.setItem('gemini-api-key', apiKeyInput.value.trim());
});

// --- Init ---
// Lazy-loaded modules (populated in init)
let initScene, WidowXController, detectAndPlan, planToWaypoints;
let eulerToRotationMatrix, rotationMatrixTo6d;

async function init() {
  try {
    // Load modules sequentially to avoid HTTP/1.1 connection saturation
    log('Loading modules...', 'info');
    const sceneModule = await import('./mujoco-scene.js');
    initScene = sceneModule.initScene;
    const ikModule = await import('./ik-solver.js');
    WidowXController = ikModule.WidowXController;
    const geminiModule = await import('./gemini-er.js');
    detectAndPlan = geminiModule.detectAndPlan;
    planToWaypoints = geminiModule.planToWaypoints;
    const mathModule = await import('./math-utils.js');
    eulerToRotationMatrix = mathModule.eulerToRotationMatrix;
    rotationMatrixTo6d = mathModule.rotationMatrixTo6d;
    log('Modules loaded.', 'info');

    mujocoScene = await initScene(
      document.getElementById('viewer'),
      (msg) => {
        loadingText.textContent = msg;
        log(msg, 'info');
      },
    );

    ikController = new WidowXController(mujocoScene.mj, mujocoScene.model);

    mujocoScene.setVisibleBodies(null); // show all

    // Start in free camera mode
    chkFreeCam.checked = true;
    mujocoScene.setFreeCam(true);

    loadingOverlay.style.display = 'none';
    statusSim.textContent = 'Sim: ready';
    btnRun.disabled = false;
    log('Scene loaded. Enter API key and click Run, or use cached plan.', 'success');
  } catch (err) {
    loadingText.textContent = `Error: ${err.message}`;
    log(`Init failed: ${err.message}`, 'error');
    log(`Stack: ${err.stack}`, 'error');
    console.error(err);
  }
}

// --- Build a default 10D action from a waypoint (xyz + default rotation + gripper) ---
function waypointToAction10d(wp) {
  // Use a default downward-facing gripper orientation
  const rot = eulerToRotationMatrix(0, Math.PI / 2, 0);
  const rot6d = rotationMatrixTo6d(rot);
  return new Float64Array([
    wp.xyz[0], wp.xyz[1], wp.xyz[2],
    rot6d[0], rot6d[1], rot6d[2],
    rot6d[3], rot6d[4], rot6d[5],
    wp.gripper,
  ]);
}

// --- Update waypoint sidebar ---
function updateWaypointUI() {
  waypointList.innerHTML = '';
  for (let i = 0; i < waypoints.length; i++) {
    const div = document.createElement('div');
    div.className = 'wp-item' + (i === currentWpIdx ? ' active' : i < currentWpIdx ? ' done' : '');
    const wp = waypoints[i];
    const grip = wp.gripper > 0.5 ? 'open' : 'close';
    div.textContent = `${i}: (${wp.xyz[0].toFixed(3)}, ${wp.xyz[1].toFixed(3)}, ${wp.xyz[2].toFixed(3)}) ${grip}`;
    waypointList.appendChild(div);
  }
}

// --- Animation loop ---
function animate() {
  if (!running || !mujocoScene) return;

  if (currentWpIdx < waypoints.length) {
    const wp = waypoints[currentWpIdx];
    const action = waypointToAction10d(wp);
    const ctrlTarget = ikController.solveIk(mujocoScene.data.qpos, action);

    if (ctrlTarget) {
      WidowXController.applyControl(mujocoScene.data.ctrl, ctrlTarget);
    }

    for (let i = 0; i < PHYSICS_STEPS_PER_FRAME; i++) {
      mujocoScene.step();
      simStep++;
      wpSteps++;
    }

    const dwell = wp.gripperChange ? STEPS_GRIPPER_CHANGE : STEPS_PER_WAYPOINT;
    if (wpSteps >= dwell) {
      wpSteps = 0;
      currentWpIdx++;
      updateWaypointUI();
      if (currentWpIdx < waypoints.length) {
        log(`Waypoint ${currentWpIdx}/${waypoints.length}`, 'info');
      }
    }
  } else {
    // Done with all waypoints
    for (let i = 0; i < PHYSICS_STEPS_PER_FRAME; i++) {
      mujocoScene.step();
      simStep++;
    }
    if (!running) return; // already stopped
    running = false;
    log('Pick-and-place complete!', 'success');
    btnRun.disabled = false;
    btnPrebaked.disabled = false;
  }

  // Update visuals
  mujocoScene.updateVisuals();
  mujocoScene.updateWaypointMarkers(waypoints, currentWpIdx);
  mujocoScene.controls.update();
  mujocoScene.render();

  // Status bar
  statusStep.textContent = `Step: ${simStep}`;
  const d = mujocoScene.data;
  const lfId = mujocoScene.model.body('wx250s/left_finger_link').id;
  const rfId = mujocoScene.model.body('wx250s/right_finger_link').id;
  const ex = (d.xpos[lfId * 3] + d.xpos[rfId * 3]) / 2;
  const ey = (d.xpos[lfId * 3 + 1] + d.xpos[rfId * 3 + 1]) / 2;
  const ez = (d.xpos[lfId * 3 + 2] + d.xpos[rfId * 3 + 2]) / 2;
  statusEe.textContent = `EE: (${ex.toFixed(3)}, ${ey.toFixed(3)}, ${ez.toFixed(3)})`;

  animationId = requestAnimationFrame(animate);
}

// --- Run pipeline ---
async function runPipeline(useApiKey) {
  if (!mujocoScene) return;

  btnRun.disabled = true;
  btnPrebaked.disabled = true;
  running = false;
  if (animationId) cancelAnimationFrame(animationId);

  try {
    // Snap to primary camera, grab the frame, then restore the free cam view
    log('Grabbing primary camera image...', 'info');
    mujocoScene.updateObstacleMarkers([]);
    mujocoScene.updateWaypointMarkers([], 0);
    mujocoScene.updateVisuals();
    mujocoScene.renderPrimaryCamera();
    const imageBase64 = mujocoScene.capturePrimaryImage();
    const depthBuffer = mujocoScene.capturePrimaryDepthBuffer();
    mujocoScene.restoreFreeCam();

    const img = document.createElement('img');
    img.src = 'data:image/jpeg;base64,' + imageBase64;
    img.style.maxWidth = '100%';
    logDiv.appendChild(img);

    // Visualize depth buffer as grayscale (near=bright, far=dark), clipped to 2m
    {
      const { data, width, height } = depthBuffer;
      const CLIP = 2.0; // metres — clip far background
      const canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      canvas.style.maxWidth = '100%';
      const ctx = canvas.getContext('2d');
      const imgData = ctx.createImageData(width, height);
      for (let i = 0; i < width * height; i++) {
        const d = Math.min(data[i], CLIP);
        const v = Math.round((1 - d / CLIP) * 255); // near=255, far=0
        imgData.data[i * 4 + 0] = v;
        imgData.data[i * 4 + 1] = v;
        imgData.data[i * 4 + 2] = v;
        imgData.data[i * 4 + 3] = 255;
      }
      ctx.putImageData(imgData, 0, 0);
      logDiv.appendChild(canvas);
    }

    logDiv.scrollTop = logDiv.scrollHeight;

    const apiKey = useApiKey ? apiKeyInput.value.trim() : null;
    const task = taskInput.value.trim() || undefined;
    const { camPos, camRot, fovyDeg } = mujocoScene.getPrimaryCameraParams();
    const { detections, planSteps, obstacles } = await detectAndPlan(
      apiKey, imageBase64, log, { camPos, camRot, fovyDeg, depthBuffer }, task,
    );

    log(`Detections: ${JSON.stringify(detections)}`, 'info');
    log(`Plan: ${planSteps.length} steps`, 'info');

    // Visualize 3D obstacles
    mujocoScene.updateObstacleMarkers(obstacles);

    // Convert plan to 3D waypoints
    waypoints = planToWaypoints(planSteps, camPos, camRot, fovyDeg);
    currentWpIdx = 0;
    simStep = 0;
    wpSteps = 0;

    log(`Generated ${waypoints.length} waypoints`, 'success');
    updateWaypointUI();
    mujocoScene.updateWaypointMarkers(waypoints, 0);
    mujocoScene.render();

    // Start animation
    running = true;
    statusSim.textContent = 'Sim: running';
    animate();

  } catch (err) {
    log(`Pipeline error: ${err.message}`, 'error');
    console.error(err);
    btnRun.disabled = false;
    btnPrebaked.disabled = false;
  }
}

// --- UI wiring ---
btnRun.addEventListener('click', () => {
  if (!apiKeyInput.value.trim()) {
    log('Please enter a Gemini API key.', 'warn');
    return;
  }
  runPipeline(true);
});

btnPrebaked.addEventListener('click', () => {
  const refreshText = 'Refresh to reset';
  if (btnPrebaked.textContent !== refreshText) {
    btnPrebaked.textContent = refreshText;
    runPipeline(false);
  } else {
    location.reload();
  }
});

chkFreeCam.addEventListener('change', () => {
  if (mujocoScene) {
    mujocoScene.setFreeCam(chkFreeCam.checked);
    mujocoScene.render();
  }
});

// Start
init();
