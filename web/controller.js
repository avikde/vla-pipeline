/**
 * WidowX velocity controller — single damped-pinv step per animation frame.
 *
 * Maintains an internal joint reference q. Each call advances q one step
 * toward the target; the animation loop provides the iteration.
 */

import {
  vec3, sub, clamp, ee6dToPosRot, gripperActionToCtrl,
  matMul, matTranspose, matEye, matSolve,
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
    damping = 1e-4,
    useOrientation = true,
  } = {}) {
    this._mj = mj;
    this._model = model;
    this._damping = damping;
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

    // Home joint positions for null-space regularization
    const homeCtrl = model.key('home').ctrl;
    this._homeQ = new Float64Array(6);
    for (let i = 0; i < 6; i++) this._homeQ[i] = homeCtrl[i];

    this._nv = model.nv;

    // Pre-allocate Jacobian buffers (mj_jacBody requires WASM-owned memory)
    this._jacpBuf = new mj.DoubleBuffer(3 * this._nv);
    this._jacrBuf = new mj.DoubleBuffer(3 * this._nv);

    // Internal joint reference — initialized on first solveIk call or reset()
    this._q = null;
  }

  /** Reset internal joint reference from actual qpos (call before each new run). */
  reset(qpos) {
    this._q = new Float64Array(6);
    for (let i = 0; i < 6; i++) this._q[i] = qpos[this._qposAddrs[i]];
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
   * Advance internal q one step toward the target EE pose.
   * Returns Float32Array(7) ctrl target.
   *
   * @param {Float64Array|Float32Array} qpos - Current generalized positions (for lazy init)
   * @param {Float64Array|Float32Array} action10d - [xyz(3), rot6d(6), gripper(1)]
   * @returns {Float32Array} ctrl_target [6 joints + 1 gripper]
   */
  calcPosTarget(qpos, action10d) {
    if (!this._q) this.reset(qpos);

    const { pos: targetPos, rot: targetRot } = ee6dToPosRot(action10d);
    const gripperVal = action10d[9];

    const scratch = this._scratch;
    // Set scratch from internal _q (not actual qpos — we drive _q as the reference)
    for (let i = 0; i < qpos.length; i++) scratch.qpos[i] = qpos[i];
    for (let i = 0; i < 6; i++) scratch.qpos[this._qposAddrs[i]] = this._q[i];
    for (let i = 0; i < scratch.qvel.length; i++) scratch.qvel[i] = 0;

    this._mj.mj_forward(this._model, scratch);

    const eePos = this._eePos(scratch);
    const posErr = sub(targetPos, eePos);

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
        rotErr = vec3(
          angle * s * (m3get(Rerr, 2, 1) - m3get(Rerr, 1, 2)),
          angle * s * (m3get(Rerr, 0, 2) - m3get(Rerr, 2, 0)),
          angle * s * (m3get(Rerr, 1, 0) - m3get(Rerr, 0, 1)),
        );
      } else {
        rotErr = vec3(0, 0, 0);
      }

      const ORI_WEIGHT = 0.2;
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

    // Damped pseudo-inverse: dq = J^T (J J^T + lambda I)^{-1} err
    const Jt = matTranspose(J, m, nArm);
    const JJt = matMul(J, m, nArm, Jt, m);
    const lambdaI = matEye(m);
    for (let i = 0; i < m * m; i++) lambdaI[i] *= this._damping;
    const JJtDamped = new Float64Array(m * m);
    for (let i = 0; i < m * m; i++) JJtDamped[i] = JJt[i] + lambdaI[i];
    const Im = matEye(m);
    const JJtInv = matSolve(JJtDamped, m, Im, m);
    const Jpinv = matMul(Jt, nArm, m, JJtInv, m);

    const dq = new Float64Array(nArm);
    for (let i = 0; i < nArm; i++) {
      let s = 0;
      for (let j = 0; j < m; j++) s += Jpinv[i * m + j] * err[j];
      dq[i] = s;
    }

    // Home bias (null-space regularization)
    const HOME_BIAS = 0.02;
    for (let i = 0; i < nArm; i++)
      dq[i] += HOME_BIAS * (this._homeQ[i] - this._q[i]);

    // Clamp step size
    const MAX_DQ = 0.1;
    let dqMax = 0;
    for (let i = 0; i < nArm; i++) dqMax = Math.max(dqMax, Math.abs(dq[i]));
    if (dqMax > MAX_DQ) {
      const s = MAX_DQ / dqMax;
      for (let i = 0; i < nArm; i++) dq[i] *= s;
    }

    // Advance internal joint reference and clamp to joint limits
    for (let i = 0; i < nArm; i++) {
      this._q[i] += dq[i];
      const jid = this._jntIds[i];
      this._q[i] = clamp(this._q[i], this._model.jnt_range[jid * 2], this._model.jnt_range[jid * 2 + 1]);
    }

    const ctrlTarget = new Float32Array(7);
    for (let i = 0; i < 6; i++) ctrlTarget[i] = this._q[i];
    ctrlTarget[6] = gripperActionToCtrl(gripperVal);
    return ctrlTarget;
  }

  /** Write ctrl_target into the data.ctrl array. */
  static applyControl(ctrl, ctrlTarget) {
    for (let i = 0; i < 7; i++) ctrl[i] = ctrlTarget[i];
  }
}
