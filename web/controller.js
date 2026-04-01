/**
 * WidowX velocity controller — single damped-pinv step per animation frame.
 *
 * Each call computes dq from current qpos and returns qpos + dq as the
 * position control target; the animation loop provides the iteration.
 */

import {
  vec3, sub, scale, norm, clamp, ee6dToPosRot, gripperActionToCtrl,
  matMul, matTranspose, matLeastSquares,
  m3get, m3trace, m3mul, m3transpose,
} from './math-utils.js';

// Constants matching widowx_control.py
const ARM_JOINTS = ['waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate'];
const EE_BODY = 'wx250s/gripper_link';
const LEFT_FINGER_BODY = 'wx250s/left_finger_link';
const RIGHT_FINGER_BODY = 'wx250s/right_finger_link';

export class Controller {
  /**
   * @param {object} mj - MuJoCo WASM module
   * @param {object} model - MjModel
   * @param {object} options
   */
  constructor(mj, model, {
    useOrientation = true,
  } = {}) {
    this._mj = mj;
    this._model = model;
    this._useOrientation = useOrientation;

    // Allocate scratch data
    this._scratch = new mj.MjData(model);

    // Cache body IDs
    this._eeId = model.body(EE_BODY).id;
    this._lfId = model.body(LEFT_FINGER_BODY).id;
    this._rfId = model.body(RIGHT_FINGER_BODY).id;

    // Cache joint IDs and addresses
    this._jntIds = ARM_JOINTS.map(n => model.jnt(n).id);
    this._qposAddrs = this._jntIds.map(j => model.jnt_qposadr[j]);
    this._dofAddrs = this._jntIds.map(j => model.jnt_dofadr[j]);

    this._nv = model.nv;

    // Pre-allocate Jacobian buffers (mj_jacBody requires WASM-owned memory)
    this._jacpBuf = new mj.DoubleBuffer(3 * this._nv);
    this._jacrBuf = new mj.DoubleBuffer(3 * this._nv);
  }

  /** Compute EE position: finger midpoint. */
  _eePos(d) {
    const lf = this._lfId * 3;
    const rf = this._rfId * 3;
    return vec3(
      (d.xpos[lf] + d.xpos[rf]) / 2,
      (d.xpos[lf + 1] + d.xpos[rf + 1]) / 2,
      (d.xpos[lf + 2] + d.xpos[rf + 2]) / 2,
    );
  }

  /**
   * Compute EE-space repulsive gradient from obstacles: -∂b/∂p (3D vector).
   * Caller maps to joint space via Jp^+.
   *
   * @param {object} scratch - MjData with fwd kinematics already run
   * @param {Array} obstacles - [{center, horizontal_size, vertical_size, type}, ...]
   * @returns {Float64Array} g_p (3) — repulsive direction in EE position space
   */
  _obstacleGradient(scratch, obstacles) {
    const OBST_GAIN = 0.0002;
    const OBST_CUTOFF = 1.5; // normalized distance (1 = surface); skip if farther
    const EPSILON = 0.01;    // clamps rSafe away from zero at the surface

    const g_p = new Float64Array(3);
    const eePos = this._eePos(scratch);

    this.lastObstacleGradients = [];

    for (const obs of obstacles) {
      const { center, horizontal_size: hs, vertical_size: vs, type } = obs;
      if (type !== 'obstacle') continue;
      if (!center || !hs || !vs) continue;

      const dx = eePos[0] - center[0];
      const dy = eePos[1] - center[1];
      const dz = eePos[2] - center[2];

      // r=1 on ellipsoid surface, r<1 inside, r>1 outside
      const r = Math.sqrt((dx / hs) ** 2 + (dy / hs) ** 2 + (dz / vs) ** 2);

      const g_p_obs = new Float64Array(3);
      if (r <= OBST_CUTOFF && r >= 1e-9) {
        const rSafe = Math.max(r - 1.0, EPSILON);
        const b = 1.0 / (rSafe * rSafe);

        // -∂b/∂p_x = +2b/(rSafe·r) · (dx/hs²)  (repels: positive when dx>0)
        const sc = OBST_GAIN * 2.0 * b / (rSafe * r);
        g_p_obs[0] = sc * (dx / (hs * hs));
        g_p_obs[1] = sc * (dy / (hs * hs));
        g_p_obs[2] = sc * (dz / (vs * vs));
        g_p[0] += g_p_obs[0];
        g_p[1] += g_p_obs[1];
        g_p[2] += g_p_obs[2];
      }
      this.lastObstacleGradients.push({ center: new Float64Array(center), g_p: g_p_obs });
    }
    return g_p;
  }

  /**
   * Compute one vel-control step from current qpos toward the target EE pose.
   * Returns Float32Array(7) ctrl target = qpos + dq.
   *
   * @param {Float64Array|Float32Array} qpos - Current generalized positions
   * @param {Float64Array|Float32Array} action10d - [xyz(3), rot6d(6), gripper(1)]
   * @param {Array} obstacles - optional obstacle array for avoidance
   * @returns {Float32Array} ctrl_target [6 joints + 1 gripper]
   */
  calcPosTarget(qpos, action10d, obstacles = []) {
    const MAX_POS_ERR = 0.2; // m — caps dq magnitude without explicit clamping
    const MAX_ROT_ERR = 1.0; // rad — analogous to MAX_POS_ERR
    const ORI_WEIGHT = 1.5; // scale orientation [rad] err vs. pos [m]
    const MAX_DQ = 0.3; // rad/frame for each joint

    const { pos: targetPos, rot: targetRot } = ee6dToPosRot(action10d);
    const gripperVal = action10d[9];

    const scratch = this._scratch;
    for (let i = 0; i < qpos.length; i++) scratch.qpos[i] = qpos[i];
    for (let i = 0; i < scratch.qvel.length; i++) scratch.qvel[i] = 0;

    this._mj.mj_forward(this._model, scratch);

    const eePos = this._eePos(scratch);
    const rawPosErr = sub(targetPos, eePos);
    const posErrNorm = norm(rawPosErr);
    const posErr = posErrNorm > MAX_POS_ERR ? scale(rawPosErr, MAX_POS_ERR / posErrNorm) : rawPosErr;

    this._mj.mj_jacBody(this._model, scratch, this._jacpBuf, this._jacrBuf, this._eeId);
    const jacpFull = this._jacpBuf.GetView();
    const jacrFull = this._jacrBuf.GetView();

    const nv = this._nv;
    const nArm = 6;

    const Jp = new Float64Array(3 * nArm);
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < nArm; j++)
        Jp[i * nArm + j] = jacpFull[i * nv + this._dofAddrs[j]];

    let J, err, m;
    if (this._useOrientation) {
      const Jr = new Float64Array(3 * nArm);
      for (let i = 0; i < 3; i++)
        for (let j = 0; j < nArm; j++)
          Jr[i * nArm + j] = jacrFull[i * nv + this._dofAddrs[j]];

      const eeOff = this._eeId * 9;
      const Rcurr = new Float64Array(9);
      for (let i = 0; i < 9; i++) Rcurr[i] = scratch.xmat[eeOff + i];

      const Rerr = m3mul(targetRot, m3transpose(Rcurr));
      const trace = clamp((m3trace(Rerr) - 1.0) / 2.0, -1, 1);
      const angle = Math.acos(trace);

      let rotErr;
      if (angle > 1e-6) {
        const s = 1.0 / (2.0 * Math.sin(angle));
        const rawRotErr = vec3(
          angle * s * (m3get(Rerr, 2, 1) - m3get(Rerr, 1, 2)),
          angle * s * (m3get(Rerr, 0, 2) - m3get(Rerr, 2, 0)),
          angle * s * (m3get(Rerr, 1, 0) - m3get(Rerr, 0, 1)),
        );
        const rotErrNorm = norm(rawRotErr);
        rotErr = rotErrNorm > MAX_ROT_ERR
          ? scale(rawRotErr, MAX_ROT_ERR / rotErrNorm)
          : rawRotErr;
      } else {
        rotErr = vec3(0, 0, 0);
      }

      m = 6;
      J = new Float64Array(6 * nArm);
      J.set(Jp, 0);
      J.set(Jr, 3 * nArm);
      err = new Float64Array(6);
      err[0] = posErr[0]; err[1] = posErr[1]; err[2] = posErr[2];
      err[3] = ORI_WEIGHT * rotErr[0];
      err[4] = ORI_WEIGHT * rotErr[1];
      err[5] = ORI_WEIGHT * rotErr[2];
    } else {
      m = 3;
      J = Jp;
      err = new Float64Array([posErr[0], posErr[1], posErr[2]]);
    }

    const dq_task = matLeastSquares(J, m, nArm, err);
    // Map EE-space repulsive gradient to joint space via Jp^+ (IK step),
    // not Jp^T — we want the joint motion that produces the EE displacement.
    const g_p = this._obstacleGradient(scratch, obstacles);
    const Jo = matLeastSquares(Jp, 3, nArm, g_p);
    const dq = new Float64Array(nArm);
    for (let i = 0; i < nArm; i++) dq[i] = dq_task[i] + Jo[i];

    // Clamp total step size to limit arm speed
    const dqNorm = Math.sqrt(dq.reduce((s, v) => s + v * v, 0));
    if (dqNorm > MAX_DQ) {
      const dqScale = MAX_DQ / dqNorm;
      for (let i = 0; i < nArm; i++) dq[i] *= dqScale;
    }

    // qpos + dq, clamped to joint limits
    const ctrlTarget = new Float32Array(7);
    for (let i = 0; i < nArm; i++) {
      const jid = this._jntIds[i];
      ctrlTarget[i] = clamp(
        qpos[this._qposAddrs[i]] + dq[i],
        this._model.jnt_range[jid * 2],
        this._model.jnt_range[jid * 2 + 1],
      );
    }
    ctrlTarget[6] = gripperActionToCtrl(gripperVal);
    return ctrlTarget;
  }

  /** Write ctrl_target into the data.ctrl array. */
  static applyControl(ctrl, ctrlTarget) {
    for (let i = 0; i < 7; i++) ctrl[i] = ctrlTarget[i];
  }
}
