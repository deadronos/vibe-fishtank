/** An orbit camera with damping and a gentle drift when left alone. */

import {
  clamp,
  cross,
  invert,
  lookAt,
  multiply,
  normalize,
  perspective,
  sub,
  transformPoint,
  type Floats,
  type Vec3,
} from "./lib/math";

const FOV = (34 * Math.PI) / 180;
const TARGET: Vec3 = [0, 0.62, 0];
const MIN_RADIUS = 2.3;
const MAX_RADIUS = 7;
const IDLE_AFTER = 7;
/** The tank plus a margin, used to keep it in frame on narrow screens. */
const FIT_WIDTH = 4;

export class OrbitCamera {
  position: Vec3 = [0, 0, 1];
  right: Vec3 = [1, 0, 0];
  up: Vec3 = [0, 1, 0];
  viewProj: Floats = new Float32Array(16);
  invViewProj: Floats = new Float32Array(16);

  private yaw = -0.16;
  private pitch = 0.07;
  private zoom = 1;
  private targetYaw = -0.16;
  private targetPitch = 0.07;
  private targetZoom = 1;
  private idle = 0;
  private time = 0;
  private readonly reducedMotion: boolean;

  constructor(reducedMotion: boolean) {
    this.reducedMotion = reducedMotion;
  }

  orbit(dx: number, dy: number): void {
    this.targetYaw = clamp(this.targetYaw - dx * 0.0052, -1.2, 1.2);
    this.targetPitch = clamp(this.targetPitch + dy * 0.0042, -0.06, 0.55);
    this.idle = 0;
  }

  dolly(factor: number): void {
    this.targetZoom = clamp(this.targetZoom * factor, 0.55, 1.6);
    this.idle = 0;
  }

  update(dt: number, aspect: number): void {
    this.time += dt;
    this.idle += dt;
    const k = 1 - Math.exp(-dt * 7);
    this.yaw += (this.targetYaw - this.yaw) * k;
    this.pitch += (this.targetPitch - this.pitch) * k;
    this.zoom += (this.targetZoom - this.zoom) * k;

    let driftYaw = 0;
    let driftPitch = 0;
    if (!this.reducedMotion) {
      const amount = clamp((this.idle - IDLE_AFTER) / 6, 0, 1);
      driftYaw = Math.sin(this.time * 0.07) * 0.1 * amount;
      driftPitch = Math.sin(this.time * 0.05 + 1) * 0.022 * amount;
    }
    const yaw = this.yaw + driftYaw;
    const pitch = this.pitch + driftPitch;

    const halfTan = Math.tan(FOV / 2);
    const fitWidth = FIT_WIDTH / (2 * halfTan * aspect);
    const fitHeight = 2.15 / (2 * halfTan);
    const radius = clamp(
      Math.max(fitWidth, fitHeight) * this.zoom + 0.65,
      MIN_RADIUS,
      MAX_RADIUS * 1.6,
    );

    const c = Math.cos(pitch);
    this.position = [
      TARGET[0] + Math.sin(yaw) * c * radius,
      TARGET[1] + Math.sin(pitch) * radius,
      TARGET[2] + Math.cos(yaw) * c * radius,
    ];
    const forward = normalize(sub(TARGET, this.position));
    this.right = normalize(cross(forward, [0, 1, 0]));
    this.up = cross(this.right, forward);
    const view = lookAt(this.position, TARGET, [0, 1, 0]);
    this.viewProj = multiply(perspective(FOV, aspect, 0.1, 40), view);
    this.invViewProj = invert(this.viewProj);
  }

  /** A world-space ray through normalized device coordinates. */
  ray(ndcX: number, ndcY: number): Vec3 {
    return normalize(sub(transformPoint(this.invViewProj, [ndcX, ndcY, 1]), this.position));
  }
}
