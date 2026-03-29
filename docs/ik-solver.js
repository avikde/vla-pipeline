/**
 * WidowX IK solver — port of widowx_control.py WidowXController.
 *
 * Damped least-squares Jacobian IK with home-configuration regularization.
 * Uses MuJoCo WASM's mj_forward / mj_jacBody for kinematics.
 */

import {
  vec3, sub, add, scale, norm, clamp, ee6dToPosRot, gripperActionToCtrl,
  matMul, matTranspose, matEye, matSolve, vecNorm,
  m3get, m3trace, m3mul, m3transpose, m3fromCols, rotationMatrixToEuler,
} from './math-utils.js';

// Constants matching widowx_control.py
const ARM_JOINTS = ['waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate'];
const EE_BODY = 'wx250s/gripper_link';
const LEFT_FINGER_BODY = 'wx250s/left_finger_link';
const RIGHT_FINGER_BODY = 'wx250s/right_finger_link';
const FINGER_TIP_OFFSET = 0.0;

export class WidowXController {
  /**
   * @param {object} mj - MuJoCo WASM module
   * @param {object} model - MjModel
   * @param {object} options
   */
  constructor(mj, model, {
    maxIter = 120,
    tol = 1e-4,
    damping = 1e-4,
    useOrientation = true,
  } = {}) {
    this._mj = mj;
    this._model = model;
    this._maxIter = maxIter;
    this._tol = tol;
    this._damping = damping;
    this._useOrientation = useOrientation;

    // Allocate scratch data
    this._scratch = new mj.MjData(model);

    // Cache body IDs
    this._eeId = model.body(EE_BODY).id;
    this._lfId = model.body(LEFT_FINGER_BODY).id;
    this._rfId = model.body(RIGHT_FINGER_BODY).id;

    // Cache joint IDs and addresses (WASM uses jnt() accessor)
    this._jntIds = ARM_JOINTS.map(n => model.jnt(n).id);
    this._qposAddrs = this._jntIds.map(j => model.jnt_qposadr[j]);
    this._dofAddrs = this._jntIds.map(j => model.jnt_dofadr[j]);

    // Home joint positions for regularization
    const homeCtrl = model.key('home').ctrl;
    this._homeQ = new Float64Array(6);
    for (let i = 0; i < 6; i++) this._homeQ[i] = homeCtrl[i];

    this._nv = model.nv;

    // Pre-allocate Jacobian buffers using DoubleBuffer (mj_jacBody requires WASM-owned memory)
    this._jacpBuf = new mj.DoubleBuffer(3 * this._nv);
    this._jacrBuf = new mj.DoubleBuffer(3 * this._nv);
  }

  /** Compute EE position: finger midpoint (matching Python get_ee_pose). */
  _eePos(d) {
    const lf = this._lfId * 3;
    const rf = this._rfId * 3;
    return vec3(
      (d.xpos[lf] + d.xpos[rf]) / 2,
      (d.xpos[lf + 1] + d.xpos[rf + 1]) / 2,
      (d.xpos[lf + 2] + d.xpos[rf + 2]) / 2,
    );
  }

  /** Get EE rotation matrix (row-major 3x3) from xmat. */
  _eeRot(d) {
    const off = this._eeId * 9;
    return new Float64Array(d.xmat.buffer, d.xmat.byteOffset + off * 8, 9);
  }

  /**
   * Solve IK for a 10D ee6d action. Returns Float32Array(7) ctrl target or null.
   *
   * @param {Float64Array|Float32Array} qpos - Current generalized positions
   * @param {Float64Array|Float32Array} action10d - [xyz(3), rot6d(6), gripper(1)]
   * @returns {Float32Array|null} ctrl_target [6 joints + 1 gripper] or null
   */
  solveIk(qpos, action10d) {
    const { pos: targetPos, rot: targetRot } = ee6dToPosRot(action10d);
    const gripperVal = action10d[9];

    const scratch = this._scratch;
    // Copy current qpos into scratch
    for (let i = 0; i < qpos.length; i++) scratch.qpos[i] = qpos[i];
    for (let i = 0; i < scratch.qvel.length; i++) scratch.qvel[i] = 0;

    const nv = this._nv;
    const nArm = 6;

    for (let iter = 0; iter < this._maxIter; iter++) {
      this._mj.mj_forward(this._model, scratch);

      const eePos = this._eePos(scratch);
      const posErr = sub(targetPos, eePos);
      if (norm(posErr) < this._tol) break;

      // Compute full Jacobian — mj_jacBody requires WASM-owned DoubleBuffer, not JS arrays
      this._mj.mj_jacBody(this._model, scratch, this._jacpBuf, this._jacrBuf, this._eeId);
      const jacpFull = this._jacpBuf.GetView();
      const jacrFull = this._jacrBuf.GetView();

      // Extract arm DOF columns: Jp is (3 x 6)
      const Jp = new Float64Array(3 * nArm);
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < nArm; j++) {
          Jp[i * nArm + j] = jacpFull[i * nv + this._dofAddrs[j]];
        }
      }

      let J, err;
      if (this._useOrientation) {
        // Rotation Jacobian (3 x 6)
        const Jr = new Float64Array(3 * nArm);
        for (let i = 0; i < 3; i++) {
          for (let j = 0; j < nArm; j++) {
            Jr[i * nArm + j] = jacrFull[i * nv + this._dofAddrs[j]];
          }
        }

        // Current rotation (copy from xmat)
        const eeOff = this._eeId * 9;
        const Rcurr = new Float64Array(9);
        for (let i = 0; i < 9; i++) Rcurr[i] = scratch.xmat[eeOff + i];

        // R_err = targetRot @ Rcurr.T
        const Rerr = m3mul(targetRot, m3transpose(Rcurr));
        const trace = clamp((m3trace(Rerr) - 1.0) / 2.0, -1, 1);
        const angle = Math.acos(trace);

        let rotErr;
        if (angle > 1e-6) {
          const s = 1.0 / (2.0 * Math.sin(angle));
          rotErr = vec3(
            angle * s * (m3get(Rerr, 2, 1) - m3get(Rerr, 1, 2)),
            angle * s * (m3get(Rerr, 0, 2) - m3get(Rerr, 2, 0)),
            angle * s * (m3get(Rerr, 1, 0) - m3get(Rerr, 0, 1)),
          );
        } else {
          rotErr = vec3(0, 0, 0);
        }

        // J = vstack(Jp, Jr) → (6 x 6)
        J = new Float64Array(6 * nArm);
        J.set(Jp, 0);
        J.set(Jr, 3 * nArm);

        // err = [posErr, ORI_WEIGHT * rotErr]
        const ORI_WEIGHT = 0.2;
        err = new Float64Array(6);
        err[0] = posErr[0]; err[1] = posErr[1]; err[2] = posErr[2];
        err[3] = ORI_WEIGHT * rotErr[0];
        err[4] = ORI_WEIGHT * rotErr[1];
        err[5] = ORI_WEIGHT * rotErr[2];
      } else {
        J = Jp;
        err = new Float64Array([posErr[0], posErr[1], posErr[2]]);
      }

      const m = this._useOrientation ? 6 : 3;

      // Damped pseudo-inverse: J^T (J J^T + lambda I)^-1
      const Jt = matTranspose(J, m, nArm);
      const JJt = matMul(J, m, nArm, Jt, m);
      const lambdaI = matEye(m);
      for (let i = 0; i < m * m; i++) lambdaI[i] *= this._damping;
      const JJtDamped = new Float64Array(m * m);
      for (let i = 0; i < m * m; i++) JJtDamped[i] = JJt[i] + lambdaI[i];

      const Im = matEye(m);
      const JJtInv = matSolve(JJtDamped, m, Im, m);
      const Jpinv = matMul(Jt, nArm, m, JJtInv, m);

      // dq = Jpinv @ err
      const dq = new Float64Array(nArm);
      for (let i = 0; i < nArm; i++) {
        let s = 0;
        for (let j = 0; j < m; j++) s += Jpinv[i * m + j] * err[j];
        dq[i] = s;
      }

      // Home bias
      const HOME_BIAS = 0.02;
      for (let i = 0; i < nArm; i++) {
        dq[i] += HOME_BIAS * (this._homeQ[i] - scratch.qpos[this._qposAddrs[i]]);
      }

      // Clamp step size
      const MAX_DQ = 0.1;
      const dqNorm = vecNorm(dq);
      if (dqNorm > MAX_DQ) {
        const s = MAX_DQ / dqNorm;
        for (let i = 0; i < nArm; i++) dq[i] *= s;
      }

      // Update qpos
      for (let i = 0; i < nArm; i++) {
        scratch.qpos[this._qposAddrs[i]] += dq[i];
      }

      // Clamp to joint limits
      for (let i = 0; i < nArm; i++) {
        const jid = this._jntIds[i];
        const lo = this._model.jnt_range[jid * 2];
        const hi = this._model.jnt_range[jid * 2 + 1];
        scratch.qpos[this._qposAddrs[i]] = clamp(scratch.qpos[this._qposAddrs[i]], lo, hi);
      }
    }

    // Extract solution
    const ctrlTarget = new Float32Array(7);
    for (let i = 0; i < 6; i++) {
      ctrlTarget[i] = scratch.qpos[this._qposAddrs[i]];
    }
    ctrlTarget[6] = gripperActionToCtrl(gripperVal);

    return ctrlTarget;
  }

  /** Write ctrl_target into the data.ctrl array. */
  static applyControl(ctrl, ctrlTarget) {
    for (let i = 0; i < 7; i++) ctrl[i] = ctrlTarget[i];
  }
}
