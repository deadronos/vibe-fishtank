/** The aquascape: where the sand rises, and where the stones and wood sit. */

import { fbm3 } from "../lib/noise";
import { clamp, smoothstep, type Vec3 } from "../lib/math";
import { TANK } from "./common";

export interface Stone {
  x: number;
  z: number;
  radius: number;
  height: number;
  seed: number;
}

/** A rising left group and a low right group, leaving open swimming room in the middle. */
export const STONES: readonly Stone[] = [
  { x: -0.82, z: -0.18, radius: 0.3, height: 0.5, seed: 1 },
  { x: -1.14, z: -0.3, radius: 0.22, height: 0.34, seed: 2 },
  { x: -0.5, z: -0.32, radius: 0.19, height: 0.27, seed: 3 },
  { x: -0.58, z: 0.08, radius: 0.13, height: 0.13, seed: 4 },
  { x: -1.1, z: 0.1, radius: 0.12, height: 0.11, seed: 5 },
  { x: 1.12, z: -0.34, radius: 0.2, height: 0.24, seed: 6 },
  { x: 0.88, z: -0.08, radius: 0.12, height: 0.11, seed: 7 },
  { x: 0.28, z: 0.26, radius: 0.08, height: 0.06, seed: 8 },
];

export const DRIFTWOOD: readonly (readonly Vec3[])[] = [
  [
    [-0.05, 0.16, -0.4],
    [0.14, 0.32, -0.34],
    [0.3, 0.52, -0.26],
    [0.5, 0.72, -0.24],
    [0.72, 0.86, -0.3],
    [0.98, 0.96, -0.34],
  ],
  [
    [0.3, 0.52, -0.26],
    [0.36, 0.72, -0.14],
    [0.32, 0.9, -0.08],
    [0.4, 1.04, -0.02],
  ],
  [
    [0.5, 0.72, -0.24],
    [0.62, 0.74, -0.08],
    [0.8, 0.72, 0.02],
  ],
  [
    [0.14, 0.32, -0.34],
    [-0.08, 0.44, -0.28],
    [-0.2, 0.62, -0.32],
  ],
];

/** Height of the substrate. It slopes up towards the back, as in a real planted tank. */
export function sandHeight(x: number, z: number): number {
  const back = smoothstep(TANK.halfZ, -TANK.halfZ, z);
  let h = 0.1 + back * 0.15 + fbm3(x * 1.3 + 7, 3, z * 1.9, 3) * 0.06 - 0.02;
  for (const stone of STONES) {
    const d = Math.hypot(x - stone.x, z - stone.z) / (stone.radius * 1.5);
    h += stone.radius * 0.1 * (1 - smoothstep(0.4, 1, d));
  }
  return h;
}

export interface Obstacle {
  center: Vec3;
  radius: number;
}

export function createObstacles(): Obstacle[] {
  const obstacles: Obstacle[] = STONES.map((stone) => ({
    center: [stone.x, sandHeight(stone.x, stone.z) + stone.height * 0.45, stone.z],
    radius: Math.max(stone.radius, stone.height * 0.75) + 0.04,
  }));
  for (const branch of DRIFTWOOD) {
    for (const point of branch) obstacles.push({ center: point, radius: 0.09 });
  }
  return obstacles;
}

/** 0 right next to a stone, 1 far away. Used for baked contact shadows and plant placement. */
export function stoneClearance(x: number, z: number): number {
  let clearance = 1;
  for (const stone of STONES) {
    const d = Math.hypot(x - stone.x, z - stone.z) / stone.radius;
    clearance = Math.min(clearance, clamp(d - 0.75, 0, 1));
  }
  return clearance;
}
