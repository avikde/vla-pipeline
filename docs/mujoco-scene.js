/**
 * MuJoCo WASM + Three.js scene integration.
 *
 * Loads the WidowX model, syncs MuJoCo visualization scene to Three.js,
 * and provides access to simulation state for the IK solver and Gemini ER.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RENDER_WIDTH, RENDER_HEIGHT } from './math-utils.js';

// MuJoCo WASM served locally (Workers require same-origin).

// Asset paths relative to docs/
const ASSET_BASE = 'widowx/';
const STL_FILES = [
  'wx250s_1_base.stl',
  'wx250s_2_shoulder.stl',
  'wx250s_3_upper_arm.stl',
  'wx250s_4_upper_forearm.stl',
  'wx250s_5_lower_forearm.stl',
  'wx250s_6_wrist.stl',
  'wx250s_7_gripper.stl',
  'wx250s_8_gripper_prop.stl',
  'wx250s_9_gripper_bar.stl',
  'wx250s_10_gripper_finger.stl',
];
const TEXTURE_FILES = ['interbotix_black.png'];
const XML_FILES = ['wx250s.xml', 'widowx_vision_scene.xml'];

/**
 * Initialize MuJoCo WASM, load the WidowX model, and set up Three.js rendering.
 *
 * @param {HTMLCanvasElement} canvas
 * @param {Function} onProgress - (message: string) => void
 * @returns {Promise<MujocoScene>}
 */
export async function initScene(canvas, onProgress = () => {}) {
  // Yield to browser between heavy operations so the page stays responsive
  const yieldToUI = () => new Promise(r => setTimeout(r, 0));

  // --- Load MuJoCo WASM first so its thread pool has time to spin up ---
  onProgress('Loading MuJoCo WASM...');
  if (!crossOriginIsolated) {
    // SharedArrayBuffer is unavailable without COOP/COEP isolation.
    // coi-serviceworker.min.js should have reloaded the page to fix this.
    throw new Error('Page is not cross-origin isolated (SharedArrayBuffer unavailable). Please reload the page.');
  }
  console.time('loadMujoco');
  const { default: loadMujoco } = await import('./mujoco.js');
  const mj = await loadMujoco();
  console.timeEnd('loadMujoco');
  onProgress('MuJoCo WASM loaded. Downloading assets...');
  await yieldToUI();

  // --- Fetch assets (Node server handles concurrency; on GitHub Pages HTTP/2 is fine) ---
  mj.FS.mkdir('/assets');

  for (let i = 0; i < STL_FILES.length; i++) {
    const f = STL_FILES[i];
    onProgress(`Loading mesh ${i + 1}/${STL_FILES.length}: ${f}`);
    await yieldToUI();
    const resp = await fetch(`${ASSET_BASE}assets/${f}`);
    if (!resp.ok) throw new Error(`Failed to fetch ${f}: ${resp.status}`);
    mj.FS.writeFile(`/assets/${f}`, new Uint8Array(await resp.arrayBuffer()));
  }

  for (const f of TEXTURE_FILES) {
    onProgress(`Loading texture: ${f}`);
    await yieldToUI();
    const resp = await fetch(`${ASSET_BASE}assets/${f}`);
    if (!resp.ok) throw new Error(`Failed to fetch ${f}: ${resp.status}`);
    mj.FS.writeFile(`/assets/${f}`, new Uint8Array(await resp.arrayBuffer()));
  }

  for (const f of XML_FILES) {
    onProgress(`Loading XML: ${f}`);
    await yieldToUI();
    const resp = await fetch(`${ASSET_BASE}${f}`);
    if (!resp.ok) throw new Error(`Failed to fetch ${f}: ${resp.status}`);
    mj.FS.writeFile(`/${f}`, await resp.text());
  }

  // Pre-allocate worker threads for MuJoCo's thread pool.
  // from_xml_path blocks the main thread while compiling the model and needs
  // worker threads ready. If they're allocated on-demand, the main thread
  // deadlocks waiting for workers that haven't loaded WASM yet.
  onProgress('Warming up thread pool...');
  await yieldToUI();
  const PThread = mj.PThread;
  const needed = 4;
  while (PThread.unusedWorkers.length < needed) {
    PThread.allocateUnusedWorker();
    PThread.loadWasmModuleToWorker(PThread.unusedWorkers[PThread.unusedWorkers.length - 1]);
  }
  await new Promise(r => setTimeout(r, 1000));

  onProgress('Creating MuJoCo model...');
  await yieldToUI();

  const model = mj.MjModel.from_xml_path('/widowx_vision_scene.xml');
  onProgress('Model created. Building MjData...');
  await yieldToUI();
  const data = new mj.MjData(model);
  console.log('MjData created OK');

  // Set home position
  const homeKey = model.key('home');
  const homeQpos = homeKey.qpos;
  const homeCtrl = homeKey.ctrl;
  for (let i = 0; i < 8; i++) data.qpos[i] = homeQpos[i];
  for (let i = 0; i < model.nu; i++) data.ctrl[i] = homeCtrl[i];
  mj.mj_forward(model, data);

  // Settle physics
  onProgress('Settling physics...');
  await yieldToUI();
  for (let i = 0; i < 100; i++) mj.mj_step(model, data);

  // Create MjvScene for visualization
  const mjvScene = new mj.MjvScene(model, 500);
  const mjvCamera = new mj.MjvCamera();
  const mjvOption = new mj.MjvOption();
  const mjvPerturb = new mj.MjvPerturb();

  onProgress('Setting up 3D viewer...');
  const scene = new MujocoScene(
    mj, model, data, mjvScene, mjvCamera, mjvOption, mjvPerturb, canvas,
  );

  onProgress('Ready');
  return scene;
}

// --- Three.js geometry creation from MuJoCo geom types ---

// MuJoCo geom type constants
const mjGEOM_PLANE = 0;
const mjGEOM_HFIELD = 1;
const mjGEOM_SPHERE = 2;
const mjGEOM_CAPSULE = 3;
const mjGEOM_ELLIPSOID = 4;
const mjGEOM_CYLINDER = 5;
const mjGEOM_BOX = 6;
const mjGEOM_MESH = 7;

function createPrimitiveGeometry(type, size) {
  switch (type) {
    case mjGEOM_PLANE:
      return new THREE.PlaneGeometry(size[0] * 2 || 10, size[1] * 2 || 10);
    case mjGEOM_SPHERE:
      return new THREE.SphereGeometry(size[0], 24, 16);
    case mjGEOM_CAPSULE:
      return new THREE.CapsuleGeometry(size[0], size[2] * 2, 12, 16).rotateX(Math.PI / 2);
    case mjGEOM_ELLIPSOID:
      return new THREE.SphereGeometry(1, 24, 16);
    case mjGEOM_CYLINDER:
      return new THREE.CylinderGeometry(size[0], size[0], size[2] * 2, 24).rotateX(Math.PI / 2);
    case mjGEOM_BOX:
      return new THREE.BoxGeometry(size[0] * 2, size[1] * 2, size[2] * 2);
    default:
      return null;
  }
}

/**
 * Build a Three.js BufferGeometry from MuJoCo model mesh data.
 * Follows the approach from zalo/mujoco_wasm mujocoUtils.js.
 */
function buildMeshGeometry(model, meshId) {
  const vertAdr = Number(model.mesh_vertadr[meshId]);
  const vertNum = Number(model.mesh_vertnum[meshId]);
  const faceAdr = Number(model.mesh_faceadr[meshId]);
  const faceNum = Number(model.mesh_facenum[meshId]);

  // Extract vertex positions
  const verts = new Float32Array(vertNum * 3);
  for (let i = 0; i < vertNum * 3; i++) {
    verts[i] = Number(model.mesh_vert[vertAdr * 3 + i]);
  }

  // Extract face indices
  const faces = new Uint32Array(faceNum * 3);
  for (let i = 0; i < faceNum * 3; i++) {
    faces[i] = Number(model.mesh_face[faceAdr * 3 + i]);
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(verts, 3));
  geometry.setIndex(new THREE.BufferAttribute(faces, 1));
  geometry.computeVertexNormals();

  return geometry;
}

/**
 * Main scene manager: owns MuJoCo state and Three.js renderer.
 */
export class MujocoScene {
  constructor(mj, model, data, mjvScene, mjvCamera, mjvOption, mjvPerturb, canvas) {
    this.mj = mj;
    this.model = model;
    this.data = data;
    this.mjvScene = mjvScene;
    this.mjvCamera = mjvCamera;
    this.mjvOption = mjvOption;
    this.mjvPerturb = mjvPerturb;

    // Three.js setup
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.setClearColor(0x4a4a4a);

    this.scene = new THREE.Scene();

    // Procedural wood texture for table (approximates MuJoCo builtin="flat" wood)
    this._woodTexture = this._buildWoodTexture();
    this._woodMatId = model.mat('wood').id;
    this._blackMatId = model.mat('black').id;

    // Camera (will be synced to MuJoCo primary camera by default)
    const aspect = canvas.clientWidth / canvas.clientHeight;
    this.camera = new THREE.PerspectiveCamera(39, aspect, 0.01, 10);
    // Default: lock to MuJoCo "primary" camera
    this._syncCameraFromMujoco('primary');

    // Orbit controls (enabled only in free-cam mode)
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.1;
    this.controls.enabled = false; // disabled by default (locked cam)

    // Lighting
    const ambient = new THREE.AmbientLight(0xffffff, 0.6);
    this.scene.add(ambient);
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(-0.3, 0.3, 1.0);
    dirLight.castShadow = true;
    this.scene.add(dirLight);
    const dirLight2 = new THREE.DirectionalLight(0xffffff, 0.5);
    dirLight2.position.set(-0.3, -0.3, 0.8);
    this.scene.add(dirLight2);

    // Geom mesh cache: maps geom index → Three.js Mesh
    this._geomMeshes = new Map();
    // Geometry cache: maps key → Three.js BufferGeometry (primitives)
    this._geomCache = new Map();
    // Mesh geometry cache: maps meshId → Three.js BufferGeometry (STL meshes)
    this._meshGeomCache = new Map();
    this._buildAllMeshGeometries();

    // Waypoint markers (Three.js spheres)
    this._waypointMarkers = [];
    // Obstacle markers (Three.js wireframe cylinders)
    this._obstacleMarkers = [];

    // Primary camera for Gemini screenshot
    this._primaryCamId = model.cam('primary').id;

    // Body filter for debugging (null = show all mesh geoms)
    this._allowedBodyIds = null;

    // Handle resize
    this._resizeObserver = new ResizeObserver(() => this._onResize());
    this._resizeObserver.observe(canvas);
    this._onResize();
  }

  /** Pre-build Three.js BufferGeometry for each MuJoCo mesh. */
  _buildAllMeshGeometries() {
    const nmesh = this.model.nmesh;
    for (let meshId = 0; meshId < nmesh; meshId++) {
      this._meshGeomCache.set(meshId, buildMeshGeometry(this.model, meshId));
    }
  }

  /** Get primary camera parameters for Gemini ER pixel-to-world projection. */
  getPrimaryCameraParams() {
    const camId = this._primaryCamId;
    const camPos = new Float64Array(3);
    const camRot = new Float64Array(9);
    const off3 = camId * 3;
    const off9 = camId * 9;
    for (let i = 0; i < 3; i++) camPos[i] = this.data.cam_xpos[off3 + i];
    for (let i = 0; i < 9; i++) camRot[i] = this.data.cam_xmat[off9 + i];
    const fovyDeg = this.model.cam_fovy[camId];
    return { camPos, camRot, fovyDeg };
  }

  /** Sync Three.js camera to a named MuJoCo camera. */
  _syncCameraFromMujoco(name) {
    const camId = this.model.cam(name).id;
    this.mj.mj_forward(this.model, this.data);

    const off3 = camId * 3;
    const off9 = camId * 9;
    const pos = this.data.cam_xpos;
    const mat = this.data.cam_xmat;

    this.camera.position.set(pos[off3], pos[off3 + 1], pos[off3 + 2]);

    // MuJoCo cam_xmat columns are camera X, Y, Z axes in world frame.
    // Camera looks along -Z in its own frame.
    // Three.js camera looks along -Z in its local frame.
    const m = new THREE.Matrix4();
    m.set(
      mat[off9 + 0], mat[off9 + 1], mat[off9 + 2], pos[off3],
      mat[off9 + 3], mat[off9 + 4], mat[off9 + 5], pos[off3 + 1],
      mat[off9 + 6], mat[off9 + 7], mat[off9 + 8], pos[off3 + 2],
      0, 0, 0, 1,
    );
    this.camera.matrixAutoUpdate = false;
    this.camera.matrix.copy(m);
    this.camera.matrixWorldNeedsUpdate = true;
    this.camera.fov = this.model.cam_fovy[camId];
    this.camera.updateProjectionMatrix();
  }

  /** Enable/disable free orbit camera. */
  setFreeCam(enabled) {
    this.controls.enabled = enabled;
    if (enabled) {
      this.camera.matrixAutoUpdate = true;
      // Scene is Z-up (MuJoCo convention)
      this.camera.up.set(0, 0, 1);
      // Over-the-shoulder position: behind (negative Y), elevated (positive Z), slightly right
      this.camera.position.set(0.5, -0.4, 0.6);
      const target = new THREE.Vector3(0.2, 0.05, 0.05);
      this.camera.lookAt(target);
      this.controls.target.copy(target);
      this.controls.update();
      // Drive controls.update() + render every frame while in free-cam mode
      const loop = () => {
        if (!this._freeCamRafId) return;
        this._freeCamRafId = requestAnimationFrame(loop);
        this.controls.update();
        this.updateVisuals();
        this.render();
      };
      this._freeCamRafId = requestAnimationFrame(loop);
    } else {
      if (this._freeCamRafId) {
        cancelAnimationFrame(this._freeCamRafId);
        this._freeCamRafId = null;
      }
      this._syncCameraFromMujoco('primary');
    }
  }

  /** Restrict mesh geom rendering to specific body names (null = show all). */
  setVisibleBodies(bodyNames) {
    if (bodyNames === null) {
      this._allowedBodyIds = null;
    } else {
      this._allowedBodyIds = new Set(bodyNames.map(n => this.model.body(n).id));
    }
  }

  /** Step MuJoCo simulation. */
  step() {
    this.mj.mj_step(this.model, this.data);
  }

  /** Update Three.js scene from MuJoCo visualization state. */
  updateVisuals() {
    // Update MjvScene
    this.mj.mjv_updateScene(
      this.model, this.data, this.mjvOption, this.mjvPerturb,
      this.mjvCamera, this.mj.mjtCatBit.mjCAT_ALL.value, this.mjvScene,
    );

    const ngeom = this.mjvScene.ngeom;

    // Hide all existing meshes first
    for (const mesh of this._geomMeshes.values()) {
      mesh.visible = false;
    }

    // Update/create meshes for each visible geom
    for (let i = 0; i < ngeom; i++) {
      const geom = this.mjvScene.geoms.get(i);
      if (!geom) continue;

      const type = geom.type;

      // Body filter for debugging (null = show all)
      if (this._allowedBodyIds !== null && type === mjGEOM_MESH) {
        const bodyId = Number(this.model.geom_bodyid[Number(geom.objid)]);
        if (!this._allowedBodyIds.has(bodyId)) continue;
      }
      // Read embind wrapper properties into plain numbers
      const s0 = Number(geom.size[0]), s1 = Number(geom.size[1]), s2 = Number(geom.size[2]);

      // Get or create Three.js mesh
      let mesh = this._geomMeshes.get(i);
      if (!mesh || mesh.userData.geomType !== type) {
        // Remove old mesh
        if (mesh) {
          this.scene.remove(mesh);
          mesh.geometry.dispose();
          mesh.material.dispose();
        }

        let geometry;
        if (type === mjGEOM_MESH) {
          // geom.objid is the model geom index; geom.dataid can differ, use objid for all lookups
          const meshId = Number(this.model.geom_dataid[Number(geom.objid)]);
          geometry = this._meshGeomCache.get(meshId);
        }
        if (!geometry) {
          const sizeArr = [s0, s1, s2];
          const geomKey = `${type}-${s0.toFixed(4)}-${s1.toFixed(4)}-${s2.toFixed(4)}-${geom.dataid}`;
          geometry = this._geomCache.get(geomKey);
          if (!geometry) {
            geometry = createPrimitiveGeometry(type, sizeArr);
            if (geometry) this._geomCache.set(geomKey, geometry);
          }
        }
        if (!geometry) continue; // skip unknown geom types

        const matid = Number(geom.matid);
        const isWood = matid === this._woodMatId;
        const isBlack = matid === this._blackMatId;
        const material = new THREE.MeshStandardMaterial({
          roughness: isWood ? 0.85 : 0.5,
          metalness: isBlack ? 0.2 : 0.0,
          ...(isWood && { map: this._woodTexture }),
          ...(isBlack && { color: new THREE.Color(0.08, 0.08, 0.08) }),
        });

        mesh = new THREE.Mesh(geometry, material);
        mesh.userData.geomType = type;
        mesh.matrixAutoUpdate = false;
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        this.scene.add(mesh);
        this._geomMeshes.set(i, mesh);
      }

      mesh.visible = true;

      // Apply rgba color only for plain geoms; material-based geoms keep their color
      if (!mesh.material.map && Number(geom.matid) !== this._blackMatId) {
        const rgba = geom.rgba;
        const r = Number(rgba[0]), g = Number(rgba[1]), b = Number(rgba[2]), a = Number(rgba[3]);
        mesh.material.color.setRGB(r, g, b);
        mesh.material.opacity = a;
        mesh.material.transparent = a < 1.0;
      }

      // Read position and rotation from embind wrappers
      const pos = geom.pos;
      const mat = geom.mat;
      const px = Number(pos[0]), py = Number(pos[1]), pz = Number(pos[2]);
      const m0 = Number(mat[0]), m1 = Number(mat[1]), m2 = Number(mat[2]);
      const m3 = Number(mat[3]), m4 = Number(mat[4]), m5 = Number(mat[5]);
      const m6 = Number(mat[6]), m7 = Number(mat[7]), m8 = Number(mat[8]);

      // Update transform: MuJoCo provides 3x3 rotation (row-major) + position
      if (type === mjGEOM_ELLIPSOID) {
        mesh.matrix.set(
          m0 * s0, m1 * s1, m2 * s2, px,
          m3 * s0, m4 * s1, m5 * s2, py,
          m6 * s0, m7 * s1, m8 * s2, pz,
          0, 0, 0, 1,
        );
      } else {
        mesh.matrix.set(
          m0, m1, m2, px,
          m3, m4, m5, py,
          m6, m7, m8, pz,
          0, 0, 0, 1,
        );
      }
      mesh.matrixWorldNeedsUpdate = true;
    }

    // Update orbit controls if enabled
    if (this.controls.enabled) {
      this.controls.update();
    }
  }

  /** Render the Three.js scene. */
  render() {
    this.renderer.render(this.scene, this.camera);
  }

  /**
   * Render one frame from the primary MuJoCo camera, regardless of free-cam
   * state. Returns immediately after capturing the frame and restores the
   * previous camera so the user sees no flicker.
   */
  renderPrimaryCamera() {
    const cam = this.camera;
    // Save OrbitControls-compatible state
    const savedPos = cam.position.clone();
    const savedQuat = cam.quaternion.clone();
    const savedUp = cam.up.clone();
    const savedFov = cam.fov;
    const savedAutoUpdate = cam.matrixAutoUpdate;

    // Snap to primary camera and render — caller must read the canvas
    // (e.g. toDataURL) before calling restoreFreeCam().
    this._syncCameraFromMujoco('primary');
    this.renderer.render(this.scene, cam);

    // Stash state so restoreFreeCam() can put it back
    this._savedCamState = {
      pos: savedPos, quat: savedQuat, up: savedUp,
      fov: savedFov, autoUpdate: savedAutoUpdate,
    };
  }

  /** Restore camera after renderPrimaryCamera() + canvas read. */
  restoreFreeCam() {
    const s = this._savedCamState;
    if (!s) return;
    const cam = this.camera;
    cam.matrixAutoUpdate = s.autoUpdate;
    cam.position.copy(s.pos);
    cam.quaternion.copy(s.quat);
    cam.up.copy(s.up);
    cam.fov = s.fov;
    cam.updateProjectionMatrix();
    cam.updateMatrixWorld(true);
    this.renderer.render(this.scene, cam);
    this._savedCamState = null;
  }

  /**
   * Capture a depth buffer from the primary camera view.
   * Must be called after renderPrimaryCamera() and before restoreFreeCam().
   *
   * Renders the scene with MeshDepthMaterial into an offscreen target and reads
   * back pixels. Returns a Float32Array of linear view-space depth in metres,
   * row-major, y=0 at the top (matching image convention).
   *
   * @returns {{ data: Float32Array, width: number, height: number }}
   */
  capturePrimaryDepthBuffer() {
    const W = RENDER_WIDTH, H = RENDER_HEIGHT;

    // Reuse render target across calls
    if (!this._depthRenderTarget) {
      this._depthRenderTarget = new THREE.WebGLRenderTarget(W, H);
      this._depthMaterial = new THREE.MeshDepthMaterial({
        depthPacking: THREE.RGBADepthPacking,
      });
    }

    // Render scene with depth material into offscreen target.
    // Do NOT change camera.aspect — the frustum must match the RGB capture so that
    // Gemini's [0,1000] pixel coords map to the same NDC positions in both images.
    const cam = this.camera;
    const prevOverride = this.scene.overrideMaterial;
    this.scene.overrideMaterial = this._depthMaterial;
    this.renderer.setRenderTarget(this._depthRenderTarget);
    this.renderer.render(this.scene, cam);
    this.renderer.setRenderTarget(null);
    this.scene.overrideMaterial = prevOverride;

    // Read raw RGBA pixels (WebGL: y=0 at bottom)
    const raw = new Uint8Array(W * H * 4);
    this.renderer.readRenderTargetPixels(this._depthRenderTarget, 0, 0, W, H, raw);

    // Decode RGBADepthPacking → NDC depth [0,1] → linear view-space depth (metres)
    // Flip y so that row 0 = top of image (matching Gemini pixel convention)
    const near = cam.near, far = cam.far;
    const data = new Float32Array(W * H);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const srcRow = H - 1 - y; // WebGL y-flip
        const i = (srcRow * W + x) * 4;
        const r = raw[i] / 255, g = raw[i + 1] / 255, b = raw[i + 2] / 255, a = raw[i + 3] / 255;
        const ndcDepth = r + g / 255 + b / 65025 + a / 16581375;
        // NDC [0,1] → linear view-space Z in metres
        data[y * W + x] = (2 * near * far) / (far + near - ndcDepth * (far - near));
      }
    }

    return { data, width: W, height: H };
  }

  /** Create/update waypoint markers. */
  updateWaypointMarkers(waypoints, currentIdx = 0) {
    // Remove excess markers
    while (this._waypointMarkers.length > waypoints.length) {
      const m = this._waypointMarkers.pop();
      this.scene.remove(m);
      m.geometry.dispose();
      m.material.dispose();
    }
    // Add missing markers
    while (this._waypointMarkers.length < waypoints.length) {
      const geo = new THREE.SphereGeometry(0.008, 12, 8);
      const mat = new THREE.MeshBasicMaterial();
      const m = new THREE.Mesh(geo, mat);
      this.scene.add(m);
      this._waypointMarkers.push(m);
    }
    // Update positions and colors
    for (let i = 0; i < waypoints.length; i++) {
      const wp = waypoints[i];
      const m = this._waypointMarkers[i];
      m.position.set(wp.xyz[0], wp.xyz[1], wp.xyz[2]);

      const isCurrent = i === currentIdx;
      const alpha = isCurrent ? 0.9 : 0.3;
      const scale = isCurrent ? 1.5 : 1.0;
      m.scale.setScalar(scale);

      if (wp.gripper > 0.5) {
        m.material.color.setRGB(0, 1, 0); // green = open
      } else {
        m.material.color.setRGB(1, 0, 0); // red = closed
      }
      m.material.opacity = alpha;
      m.material.transparent = alpha < 1.0;
    }
  }

  /** Create/update obstacle markers as spheres at detected 3D centers. */
  updateObstacleMarkers(obstacles) {
    // Remove old markers
    for (const m of this._obstacleMarkers) {
      this.scene.remove(m);
      m.geometry.dispose();
      m.material.dispose();
    }
    this._obstacleMarkers = [];
    for (const obs of obstacles) {
      const geo = new THREE.SphereGeometry(0.015, 12, 8);
      const color = obs.type === 'obstacle' ? 0xff8800 : obs.type === 'block' ? 0xff0000 : 0x0088ff;
      const mat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.7 });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.set(obs.center[0], obs.center[1], obs.center[2]);
      this.scene.add(mesh);
      this._obstacleMarkers.push(mesh);
    }
  }

  /** Handle canvas resize. */
  _onResize() {
    const canvas = this.renderer.domElement;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (w === 0 || h === 0) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  /** Build procedural wood canvas texture matching XML: rgb1=0.55,0.42,0.30 random=0.05 texrepeat=3,3 */
  _buildWoodTexture() {
    const size = 256;
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = size;
    const ctx = canvas.getContext('2d');
    const img = ctx.createImageData(size, size);
    const d = img.data;
    const base = [140, 107, 77];   // rgb1="0.55 0.42 0.30"
    const mark = [89, 64, 46];     // markrgb="0.35 0.25 0.18"
    for (let p = 0; p < size * size; p++) {
      const noise = (Math.random() - 0.5) * 26; // ±13 ≈ random=0.05 * 255
      const ismark = Math.random() < 0.05;
      const c = ismark ? mark : base;
      d[p*4]   = Math.max(0, Math.min(255, c[0] + (ismark ? 0 : noise)));
      d[p*4+1] = Math.max(0, Math.min(255, c[1] + (ismark ? 0 : noise)));
      d[p*4+2] = Math.max(0, Math.min(255, c[2] + (ismark ? 0 : noise)));
      d[p*4+3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    const tex = new THREE.CanvasTexture(canvas);
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(3, 3);  // texrepeat="3 3"
    return tex;
  }

  /** Clean up resources. */
  dispose() {
    this._resizeObserver.disconnect();
    this.controls.dispose();
    for (const mesh of this._geomMeshes.values()) {
      this.scene.remove(mesh);
      mesh.geometry.dispose();
      mesh.material.dispose();
    }
    for (const geo of this._geomCache.values()) {
      geo.dispose();
    }
    for (const m of this._waypointMarkers) {
      this.scene.remove(m);
      m.geometry.dispose();
      m.material.dispose();
    }
    for (const m of this._obstacleMarkers) {
      this.scene.remove(m);
      m.geometry.dispose();
      m.material.dispose();
    }
    this.renderer.dispose();
  }
}
