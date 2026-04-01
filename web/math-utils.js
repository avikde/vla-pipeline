/**
 * Linear algebra and rotation math utilities.
 * Port of widowx_control.py rotation math + gemini_er_policy.py pixel_to_world_3d.
 */

// --- Vector operations (3D) ---

export function vec3(x, y, z) { return new Float64Array([x, y, z]); }

export function add(a, b) { return vec3(a[0] + b[0], a[1] + b[1], a[2] + b[2]); }

export function sub(a, b) { return vec3(a[0] - b[0], a[1] - b[1], a[2] - b[2]); }

export function scale(a, s) { return vec3(a[0] * s, a[1] * s, a[2] * s); }

export function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

export function cross(a, b) {
  return vec3(
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  );
}

export function norm(a) { return Math.sqrt(dot(a, a)); }

export function normalize(a) {
  const n = norm(a);
  return n > 1e-12 ? scale(a, 1 / n) : vec3(0, 0, 0);
}

export function clamp(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }

// --- Small matrix operations (row-major) ---

/** 3x3 matrix stored as Float64Array(9), row-major. */
export function mat3Identity() {
  return new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
}

/** Get element (i, j) from row-major 3x3. */
export function m3get(m, i, j) { return m[i * 3 + j]; }

/** Set element (i, j) in row-major 3x3. */
export function m3set(m, i, j, v) { m[i * 3 + j] = v; }

/** Get column j of row-major 3x3 as vec3. */
export function m3col(m, j) {
  return vec3(m[j], m[3 + j], m[6 + j]);
}

/** Build 3x3 from three column vectors. */
export function m3fromCols(c0, c1, c2) {
  return new Float64Array([
    c0[0], c1[0], c2[0],
    c0[1], c1[1], c2[1],
    c0[2], c1[2], c2[2],
  ]);
}

/** Multiply two 3x3 matrices (row-major). */
export function m3mul(a, b) {
  const out = new Float64Array(9);
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      let s = 0;
      for (let k = 0; k < 3; k++) s += a[i * 3 + k] * b[k * 3 + j];
      out[i * 3 + j] = s;
    }
  }
  return out;
}

/** Transpose 3x3. */
export function m3transpose(m) {
  return new Float64Array([
    m[0], m[3], m[6],
    m[1], m[4], m[7],
    m[2], m[5], m[8],
  ]);
}

/** Multiply 3x3 matrix by vec3 → vec3. */
export function m3mulv(m, v) {
  return vec3(
    m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
    m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
    m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
  );
}

/** Trace of 3x3. */
export function m3trace(m) { return m[0] + m[4] + m[8]; }

// --- NxN matrix operations for IK solver ---

/**
 * General matrix multiply: A (m x k) * B (k x n) = C (m x n).
 * All stored as flat Float64Array, row-major.
 */
export function matMul(a, aRows, aCols, b, bCols) {
  const out = new Float64Array(aRows * bCols);
  for (let i = 0; i < aRows; i++) {
    for (let j = 0; j < bCols; j++) {
      let s = 0;
      for (let k = 0; k < aCols; k++) {
        s += a[i * aCols + k] * b[k * bCols + j];
      }
      out[i * bCols + j] = s;
    }
  }
  return out;
}

/** Transpose (m x n) → (n x m). */
export function matTranspose(a, rows, cols) {
  const out = new Float64Array(rows * cols);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      out[j * rows + i] = a[i * cols + j];
    }
  }
  return out;
}

/** Identity matrix (n x n). */
export function matEye(n) {
  const out = new Float64Array(n * n);
  for (let i = 0; i < n; i++) out[i * n + i] = 1;
  return out;
}

/**
 * Solve A*X = B for X where A is (n x n), B is (n x m).
 * Returns X (n x m). Uses Gaussian elimination with partial pivoting.
 */
export function matSolve(A, n, B, m) {
  // Augmented matrix [A | B]
  const aug = new Float64Array(n * (n + m));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) aug[i * (n + m) + j] = A[i * n + j];
    for (let j = 0; j < m; j++) aug[i * (n + m) + n + j] = B[i * m + j];
  }
  const w = n + m;
  // Forward elimination with partial pivoting
  for (let col = 0; col < n; col++) {
    let maxVal = Math.abs(aug[col * w + col]);
    let maxRow = col;
    for (let row = col + 1; row < n; row++) {
      const v = Math.abs(aug[row * w + col]);
      if (v > maxVal) { maxVal = v; maxRow = row; }
    }
    if (maxRow !== col) {
      for (let j = 0; j < w; j++) {
        const tmp = aug[col * w + j];
        aug[col * w + j] = aug[maxRow * w + j];
        aug[maxRow * w + j] = tmp;
      }
    }
    const pivot = aug[col * w + col];
    if (Math.abs(pivot) < 1e-14) continue;
    for (let j = col; j < w; j++) aug[col * w + j] /= pivot;
    for (let row = col + 1; row < n; row++) {
      const factor = aug[row * w + col];
      for (let j = col; j < w; j++) aug[row * w + j] -= factor * aug[col * w + j];
    }
  }
  // Back substitution
  for (let col = n - 1; col >= 0; col--) {
    for (let row = col - 1; row >= 0; row--) {
      const factor = aug[row * w + col];
      for (let j = col; j < w; j++) aug[row * w + j] -= factor * aug[col * w + j];
    }
  }
  // Extract X
  const X = new Float64Array(n * m);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) X[i * m + j] = aug[i * w + n + j];
  }
  return X;
}

/** Minimum-norm least-squares solution: J^T (J J^T)^{-1} err.
 * J is (m x n), err is (m), returns dq (n). */
export function matLeastSquares(J, m, n, err) {
  const Jt = matTranspose(J, m, n);
  const JJt = matMul(J, m, n, Jt, m);
  const sol = matSolve(JJt, m, err, 1);
  return matMul(Jt, n, m, sol, 1);
}

/** Vector L2 norm for Float64Array of any length. */
export function vecNorm(v) {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i] * v[i];
  return Math.sqrt(s);
}

// --- Rotation math (port of widowx_control.py) ---

/** 3x3 rotation matrix -> (roll, pitch, yaw) XYZ Euler. */
export function rotationMatrixToEuler(m) {
  const sy = Math.sqrt(m[0] * m[0] + m[3] * m[3]); // sqrt(r00^2 + r10^2)
  let roll, pitch, yaw;
  if (sy >= 1e-6) {
    roll = Math.atan2(m[7], m[8]);   // atan2(r21, r22)
    pitch = Math.atan2(-m[6], sy);   // atan2(-r20, sy)
    yaw = Math.atan2(m[3], m[0]);    // atan2(r10, r00)
  } else {
    roll = Math.atan2(-m[5], m[4]);  // atan2(-r12, r11)
    pitch = Math.atan2(-m[6], sy);
    yaw = 0;
  }
  return [roll, pitch, yaw];
}

/** (roll, pitch, yaw) XYZ Euler -> 3x3 rotation matrix (row-major). */
export function eulerToRotationMatrix(roll, pitch, yaw) {
  const cr = Math.cos(roll), sr = Math.sin(roll);
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  // Rz * Ry * Rx, row-major
  return new Float64Array([
    cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
    sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
    -sp,     cp * sr,                cp * cr,
  ]);
}

/** 3x3 rotation matrix -> 6D representation [col0, col1]. */
export function rotationMatrixTo6d(m) {
  return new Float64Array([m[0], m[3], m[6], m[1], m[4], m[7]]);
}

/**
 * Unpack 10D ee6d action -> { pos: vec3, rot: mat3 }.
 * Gram-Schmidt orthonormalization of two column vectors from the 6D rotation.
 */
export function ee6dToPosRot(action10d) {
  const xyz = vec3(action10d[0], action10d[1], action10d[2]);
  const a1 = vec3(action10d[3], action10d[4], action10d[5]);
  const a2 = vec3(action10d[6], action10d[7], action10d[8]);

  const b1 = normalize(a1);
  let b2 = sub(a2, scale(b1, dot(b1, a2)));
  b2 = normalize(b2);
  const b3 = cross(b1, b2);

  return { pos: xyz, rot: m3fromCols(b1, b2, b3) };
}

// --- Pixel to world 3D (port of gemini_er_policy.py pixel_to_world_3d) ---

// Camera/render constants matching Python code
export const RENDER_WIDTH = 342;
export const RENDER_HEIGHT = 256;
export const VLA_WIDTH = 256;
export const VLA_HEIGHT = 256;

/**
 * Compute a world-frame ray from a pixel in VLA image space.
 *
 * @param {number} px - x pixel in VLA-sized image
 * @param {number} py - y pixel in VLA-sized image
 * @param {Float64Array} camPos - camera world position (3)
 * @param {Float64Array} camRot - camera rotation matrix (9, row-major) - columns = camera axes in world
 * @param {number} fovyDeg - camera field of view Y in degrees
 * @returns {{ origin: Float64Array, dir: Float64Array }} ray origin and normalized direction
 */
export function pixelToRay(px, py, camPos, camRot, fovyDeg) {
  const renderW = RENDER_WIDTH, renderH = RENDER_HEIGHT;
  const vlaW = VLA_WIDTH, vlaH = VLA_HEIGHT;

  // Camera intrinsics from fovy
  const fovyRad = fovyDeg * Math.PI / 180;
  const fy = (renderH / 2) / Math.tan(fovyRad / 2);
  const fx = fy; // square pixels
  const cxRender = renderW / 2;
  const cyRender = renderH / 2;

  // Convert pixel from VLA (squished) space to render space
  const uRender = px * (renderW / vlaW);
  const vRender = py * (renderH / vlaH);

  // Camera-frame ray direction (OpenGL/MuJoCo convention)
  const dCam = vec3(
    (uRender - cxRender) / fx,
    -(vRender - cyRender) / fy,
    -1.0,
  );

  // Transform to world frame
  const dir = normalize(m3mulv(camRot, dCam));
  return { origin: new Float64Array(camPos), dir };
}

/**
 * Project a pixel (in VLA image space) to 3D world coords via ray-plane intersection.
 *
 * @param {number} px - x pixel in VLA-sized image
 * @param {number} py - y pixel in VLA-sized image
 * @param {Float64Array} camPos - camera world position (3)
 * @param {Float64Array} camRot - camera rotation matrix (9, row-major) - columns = camera axes in world
 * @param {number} fovyDeg - camera field of view Y in degrees
 * @param {number} tableZ - height of table plane
 * @returns {Float64Array} world point (3)
 */
export function pixelToWorld3d(px, py, camPos, camRot, fovyDeg, tableZ = 0.02) {
  const ray = pixelToRay(px, py, camPos, camRot, fovyDeg);
  if (Math.abs(ray.dir[2]) < 1e-8) {
    return new Float64Array(camPos); // fallback
  }
  const t = (tableZ - ray.origin[2]) / ray.dir[2];
  return add(ray.origin, scale(ray.dir, t));
}

// --- Gripper mapping ---

export const GRIPPER_OPEN = 0.037;
export const GRIPPER_CLOSE = 0.01;

/** Map gripper scalar (0=closed, 1=open) to finger ctrl [m]. */
export function gripperActionToCtrl(gripperVal) {
  return clamp(
    GRIPPER_CLOSE + gripperVal * (GRIPPER_OPEN - GRIPPER_CLOSE),
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
  );
}
