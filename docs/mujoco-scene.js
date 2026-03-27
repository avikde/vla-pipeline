/**
 * MuJoCo WASM + Three.js scene integration.
 *
 * Loads the WidowX model, syncs MuJoCo visualization scene to Three.js,
 * and provides access to simulation state for the IK solver and Gemini ER.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// CDN URL for the MuJoCo WASM package
const MUJOCO_CDN = 'https://cdn.jsdelivr.net/npm/@mujoco/mujoco@3.6.1';

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
  onProgress('Loading MuJoCo WASM module...');

  // Dynamically import the MuJoCo WASM module from CDN.
  // The WASM module needs locateFile to find its .wasm binary on the CDN.
  const { default: loadMujoco } = await import(
    /* webpackIgnore: true */
    `${MUJOCO_CDN}/mujoco.js`
  );

  const mj = await loadMujoco({
    locateFile: (path) => `${MUJOCO_CDN}/${path}`,
  });
  onProgress('MuJoCo WASM loaded');

  // Write asset files to the emscripten virtual filesystem
  onProgress('Loading robot model assets...');
  mj.FS.mkdir('/assets');

  // Load STL meshes
  for (const stl of STL_FILES) {
    onProgress(`Loading ${stl}...`);
    const resp = await fetch(`${ASSET_BASE}assets/${stl}`);
    const buf = await resp.arrayBuffer();
    mj.FS.writeFile(`/assets/${stl}`, new Uint8Array(buf));
  }

  // Load textures
  for (const tex of TEXTURE_FILES) {
    const resp = await fetch(`${ASSET_BASE}assets/${tex}`);
    const buf = await resp.arrayBuffer();
    mj.FS.writeFile(`/assets/${tex}`, new Uint8Array(buf));
  }

  // Load XML files into VFS (needed for <include> resolution)
  for (const xml of XML_FILES) {
    const resp = await fetch(`${ASSET_BASE}${xml}`);
    const text = await resp.text();
    mj.FS.writeFile(`/${xml}`, text);
  }

  onProgress('Creating MuJoCo model...');
  // Use from_xml_path so MuJoCo resolves <include> and meshdir relative to /
  const model = mj.MjModel.from_xml_path('/widowx_vision_scene.xml');
  const data = new mj.MjData(model);

  // Set home position
  const homeKey = model.key('home');
  const homeQpos = homeKey.qpos;
  const homeCtrl = homeKey.ctrl;
  for (let i = 0; i < 8; i++) data.qpos[i] = homeQpos[i];
  for (let i = 0; i < model.nu; i++) data.ctrl[i] = homeCtrl[i];
  mj.mj_forward(model, data);

  // Settle physics
  onProgress('Settling physics...');
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

function createThreeGeometry(type, size) {
  switch (type) {
    case mjGEOM_PLANE:
      return new THREE.PlaneGeometry(size[0] * 2 || 10, size[1] * 2 || 10);
    case mjGEOM_SPHERE:
      return new THREE.SphereGeometry(size[0], 24, 16);
    case mjGEOM_CAPSULE:
      return new THREE.CapsuleGeometry(size[0], size[1] * 2, 12, 16);
    case mjGEOM_ELLIPSOID:
      return new THREE.SphereGeometry(1, 24, 16);
    case mjGEOM_CYLINDER:
      return new THREE.CylinderGeometry(size[0], size[0], size[1] * 2, 24);
    case mjGEOM_BOX:
      return new THREE.BoxGeometry(size[0] * 2, size[1] * 2, size[2] * 2);
    default:
      // For meshes and unknown types, use a small sphere placeholder
      return new THREE.SphereGeometry(0.01, 8, 8);
  }
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
    this.renderer.setClearColor(0x1a1a2e);

    this.scene = new THREE.Scene();

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
    // Geometry cache: maps key → Three.js BufferGeometry
    this._geomCache = new Map();

    // Waypoint markers (Three.js spheres)
    this._waypointMarkers = [];

    // Primary camera for Gemini screenshot
    this._primaryCamId = model.cam('primary').id;

    // Handle resize
    this._resizeObserver = new ResizeObserver(() => this._onResize());
    this._resizeObserver.observe(canvas);
    this._onResize();
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
      // Position camera at a reasonable orbit
      this.camera.position.set(0.5, 0.5, 0.4);
      this.camera.lookAt(0.2, 0, 0.1);
      this.controls.target.set(0.2, 0, 0.1);
    } else {
      this._syncCameraFromMujoco('primary');
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
      const size = geom.size;
      const pos = geom.pos;
      const mat = geom.mat;
      const rgba = geom.rgba;

      // Get or create Three.js mesh
      let mesh = this._geomMeshes.get(i);
      if (!mesh || mesh.userData.geomType !== type) {
        // Remove old mesh
        if (mesh) {
          this.scene.remove(mesh);
          mesh.geometry.dispose();
          mesh.material.dispose();
        }

        const geomKey = `${type}-${size[0].toFixed(4)}-${size[1].toFixed(4)}-${size[2].toFixed(4)}-${geom.dataid}`;
        let geometry = this._geomCache.get(geomKey);
        if (!geometry) {
          geometry = createThreeGeometry(type, size);
          this._geomCache.set(geomKey, geometry);
        }

        const material = new THREE.MeshStandardMaterial({
          roughness: 0.6,
          metalness: 0.1,
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

      // Update color
      mesh.material.color.setRGB(rgba[0], rgba[1], rgba[2]);
      mesh.material.opacity = rgba[3];
      mesh.material.transparent = rgba[3] < 1.0;

      // Update transform: MuJoCo provides 3x3 rotation (row-major) + position
      // Apply scale for ellipsoids
      if (type === mjGEOM_ELLIPSOID) {
        mesh.matrix.set(
          mat[0] * size[0], mat[1] * size[1], mat[2] * size[2], pos[0],
          mat[3] * size[0], mat[4] * size[1], mat[5] * size[2], pos[1],
          mat[6] * size[0], mat[7] * size[1], mat[8] * size[2], pos[2],
          0, 0, 0, 1,
        );
      } else {
        mesh.matrix.set(
          mat[0], mat[1], mat[2], pos[0],
          mat[3], mat[4], mat[5], pos[1],
          mat[6], mat[7], mat[8], pos[2],
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
    this.renderer.dispose();
  }
}
