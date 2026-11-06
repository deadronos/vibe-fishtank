/** Fish behaviour: boids per species, plus walls, obstacles, food and startles. */

import { clamp, type Vec3 } from "../lib/math";
import { createRng } from "../lib/noise";
import { TANK } from "../scene/common";
import { createObstacles, sandHeight, type Obstacle } from "../scene/layout";

export interface SpeciesDef {
  name: string;
  count: number;
  /** Body length in world units. */
  size: number;
  cruise: number;
  maxSpeed: number;
  /** Fraction of cruise speed a fish may slow to. */
  idle: number;
  cohesion: number;
  alignment: number;
  /** Radius within which same-species neighbours influence each other. */
  neighbourhood: number;
  yMin: number;
  yMax: number;
  /** Bottom dwellers hug the sand instead of using yMin and yMax. */
  bottom: boolean;
  /** Tail beats per unit of speed, plus a resting rate. */
  beatRest: number;
  beatPerSpeed: number;
}

export const SPECIES: readonly SpeciesDef[] = [
  {
    name: "neon tetras",
    count: 34,
    size: 0.105,
    cruise: 0.3,
    maxSpeed: 0.85,
    idle: 0.75,
    cohesion: 1.5,
    alignment: 2.4,
    neighbourhood: 0.42,
    yMin: 0.42,
    yMax: 1.12,
    bottom: false,
    beatRest: 9,
    beatPerSpeed: 30,
  },
  {
    name: "ember rasboras",
    count: 22,
    size: 0.08,
    cruise: 0.34,
    maxSpeed: 0.95,
    idle: 0.75,
    cohesion: 1.7,
    alignment: 2.2,
    neighbourhood: 0.36,
    yMin: 0.65,
    yMax: 1.22,
    bottom: false,
    beatRest: 11,
    beatPerSpeed: 36,
  },
  {
    name: "angelfish",
    count: 3,
    size: 0.23,
    cruise: 0.11,
    maxSpeed: 0.45,
    idle: 0.2,
    cohesion: 0.12,
    alignment: 0.25,
    neighbourhood: 0.8,
    yMin: 0.45,
    yMax: 1.05,
    bottom: false,
    beatRest: 3,
    beatPerSpeed: 12,
  },
  {
    name: "discus",
    count: 3,
    size: 0.2,
    cruise: 0.13,
    maxSpeed: 0.5,
    idle: 0.25,
    cohesion: 0.3,
    alignment: 0.3,
    neighbourhood: 0.7,
    yMin: 0.35,
    yMax: 0.95,
    bottom: false,
    beatRest: 3.5,
    beatPerSpeed: 12,
  },
  {
    name: "corydoras",
    count: 6,
    size: 0.1,
    cruise: 0.12,
    maxSpeed: 0.55,
    idle: 0.0,
    cohesion: 0.5,
    alignment: 0.3,
    neighbourhood: 0.5,
    yMin: 0,
    yMax: 0.3,
    bottom: true,
    beatRest: 6,
    beatPerSpeed: 34,
  },
];

export interface Fish {
  species: number;
  def: SpeciesDef;
  x: number;
  y: number;
  z: number;
  vx: number;
  vy: number;
  vz: number;
  /** Rendered heading and up vector; these lag velocity so turns look smooth. */
  fx: number;
  fy: number;
  fz: number;
  ux: number;
  uy: number;
  uz: number;
  turn: number;
  phase: number;
  scale: number;
  variation: number;
  seed: number;
  fed: number;
}

export interface Food {
  x: number;
  y: number;
  z: number;
  vy: number;
  drift: number;
  age: number;
}

const MARGIN = 0.14;
const FOOD_SINK = 0.12;
const FOOD_LIFE = 45;
const MAX_FOOD = 80;

export class School {
  readonly fish: Fish[] = [];
  readonly food: Food[] = [];
  private readonly obstacles: Obstacle[] = createObstacles();
  private startleTime = 0;
  private startlePoint: Vec3 = [0, 0, 0];
  private time = 0;
  private readonly rng = createRng(90210);

  constructor() {
    const { rng } = this;
    SPECIES.forEach((def, species) => {
      // Each school starts together, so the behaviour reads straight away.
      const homeX = (rng() - 0.5) * 1.6;
      const homeZ = (rng() - 0.5) * 0.5;
      for (let i = 0; i < def.count; i++) {
        const x = clamp(homeX + (rng() - 0.5) * 0.7, -1.2, 1.2);
        const z = clamp(homeZ + (rng() - 0.5) * 0.5, -0.4, 0.4);
        const y = def.bottom ? sandHeight(x, z) + 0.06 : def.yMin + rng() * (def.yMax - def.yMin);
        const angle = rng() * Math.PI * 2;
        const [vx, vz] = [Math.cos(angle) * def.cruise, Math.sin(angle) * def.cruise * 0.6];
        const l = Math.hypot(vx, vz);
        this.fish.push({
          species,
          def,
          x,
          y,
          z,
          vx,
          vy: 0,
          vz,
          fx: vx / l,
          fy: 0,
          fz: vz / l,
          ux: 0,
          uy: 1,
          uz: 0,
          turn: 0,
          phase: rng() * 6.28,
          scale: def.size * (0.86 + rng() * 0.28),
          variation: rng(),
          seed: rng() * 100,
          fed: 0,
        });
      }
    });
  }

  dropFood(x: number, z: number, amount = 7): void {
    const { rng } = this;
    for (let i = 0; i < amount && this.food.length < MAX_FOOD; i++) {
      this.food.push({
        x: clamp(x + (rng() - 0.5) * 0.14, -TANK.halfX + 0.03, TANK.halfX - 0.03),
        y: TANK.water - 0.005,
        z: clamp(z + (rng() - 0.5) * 0.14, -TANK.halfZ + 0.03, TANK.halfZ - 0.03),
        vy: 0,
        drift: rng() * 6.28,
        age: -rng() * 0.6,
      });
    }
  }

  startle(point: Vec3): void {
    this.startleTime = 1.4;
    this.startlePoint = point;
  }

  update(dt: number): void {
    this.time += dt;
    this.startleTime = Math.max(0, this.startleTime - dt);
    this.updateFood(dt);
    for (const fish of this.fish) this.steer(fish, dt);
    for (const fish of this.fish) this.integrate(fish, dt);
  }

  private updateFood(dt: number): void {
    for (let i = this.food.length - 1; i >= 0; i--) {
      const f = this.food[i];
      f.age += dt;
      if (f.age < 0) continue;
      const floor = sandHeight(f.x, f.z) + 0.006;
      // Food floats for a moment before it gets waterlogged and sinks.
      const sink = FOOD_SINK * clamp((f.age - 1.2) / 3, 0.02, 1);
      f.y = Math.max(floor, f.y - sink * dt);
      if (f.y > floor) {
        f.x += Math.sin(this.time * 0.9 + f.drift) * 0.012 * dt;
        f.z += Math.cos(this.time * 0.7 + f.drift) * 0.012 * dt;
      }
      if (f.age > FOOD_LIFE) this.food.splice(i, 1);
    }
  }

  private steer(fish: Fish, dt: number): void {
    const { def } = fish;
    let ax = 0;
    let ay = 0;
    let az = 0;

    // Flocking. Everyone keeps their distance; only their own kind align and gather.
    let sumX = 0;
    let sumY = 0;
    let sumZ = 0;
    let alignX = 0;
    let alignY = 0;
    let alignZ = 0;
    let neighbours = 0;
    for (const other of this.fish) {
      if (other === fish) continue;
      const dx = other.x - fish.x;
      const dy = other.y - fish.y;
      const dz = other.z - fish.z;
      const d2 = dx * dx + dy * dy + dz * dz;
      const personal = (fish.scale + other.scale) * 0.72;
      if (d2 < personal * personal && d2 > 1e-6) {
        const d = Math.sqrt(d2);
        const push = ((personal - d) / personal) * 3.2;
        ax -= (dx / d) * push;
        ay -= (dy / d) * push;
        az -= (dz / d) * push;
      }
      if (other.species === fish.species && d2 < def.neighbourhood * def.neighbourhood) {
        sumX += dx;
        sumY += dy;
        sumZ += dz;
        alignX += other.vx;
        alignY += other.vy;
        alignZ += other.vz;
        neighbours++;
      }
    }
    if (neighbours > 0) {
      ax += (sumX / neighbours) * def.cohesion;
      ay += (sumY / neighbours) * def.cohesion;
      az += (sumZ / neighbours) * def.cohesion;
      ax += (alignX / neighbours - fish.vx) * def.alignment;
      ay += (alignY / neighbours - fish.vy) * def.alignment;
      az += (alignZ / neighbours - fish.vz) * def.alignment;
    }

    // Wandering, so that a school never settles into a fixed orbit.
    const t = this.time * 0.31 + fish.seed;
    const wander = 0.2 * def.cruise * 5;
    ax += Math.sin(t * 1.3) * Math.cos(t * 0.47) * wander;
    az += Math.cos(t * 0.9 + 1.7) * Math.sin(t * 0.61) * wander * 0.7;
    ay += Math.sin(t * 1.9 + 3.1) * wander * 0.22;

    // Soft walls.
    const limitX = TANK.halfX - MARGIN;
    const limitZ = TANK.halfZ - MARGIN;
    const wall = 5;
    if (fish.x > limitX) ax -= (fish.x - limitX) * wall * 4;
    if (fish.x < -limitX) ax -= (fish.x + limitX) * wall * 4;
    if (fish.z > limitZ) az -= (fish.z - limitZ) * wall * 4;
    if (fish.z < -limitZ) az -= (fish.z + limitZ) * wall * 4;

    const sand = sandHeight(fish.x, fish.z);
    if (def.bottom) {
      ay += (sand + fish.scale * 0.42 - fish.y) * 14 - fish.vy * 5;
    } else {
      const floor = Math.max(def.yMin, sand + 0.09);
      if (fish.y < floor) ay += (floor - fish.y) * wall * 2;
      if (fish.y > def.yMax) ay -= (fish.y - def.yMax) * wall * 2;
      ay -= fish.vy * 0.5;
    }

    for (const obstacle of this.obstacles) {
      const dx = fish.x - obstacle.center[0];
      const dy = fish.y - obstacle.center[1];
      const dz = fish.z - obstacle.center[2];
      const range = obstacle.radius + 0.1 + fish.scale * 0.5;
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 < range * range && d2 > 1e-6) {
        const d = Math.sqrt(d2);
        const push = ((range - d) / range) * 7;
        ax += (dx / d) * push;
        ay += def.bottom ? 0 : (dy / d) * push;
        az += (dz / d) * push;
      }
    }

    // Food beats everything else except an actual scare.
    let targetSpeed = def.cruise;
    const closest = this.nearestFood(fish);
    if (closest) {
      const dx = closest.x - fish.x;
      const dy = closest.y - fish.y;
      const dz = closest.z - fish.z;
      const d = Math.hypot(dx, dy, dz) || 1;
      const eatRange = Math.max(0.03, fish.scale * 0.45);
      if (d < eatRange) {
        this.food.splice(this.food.indexOf(closest), 1);
        fish.fed = 1.5;
      } else {
        const hunger = 2.6;
        ax += (dx / d) * hunger * def.maxSpeed * 2;
        ay += (dy / d) * hunger * def.maxSpeed * 2;
        az += (dz / d) * hunger * def.maxSpeed * 2;
        targetSpeed = def.maxSpeed * 0.8;
      }
    } else {
      // Unhurried fish drift, then pick up pace again.
      const pace = 0.5 + 0.5 * Math.sin(this.time * 0.22 + fish.seed * 2.3);
      targetSpeed = def.cruise * (def.idle + (1 - def.idle) * pace * pace * 1.6);
    }

    if (this.startleTime > 0) {
      const dx = fish.x - this.startlePoint[0];
      const dy = fish.y - this.startlePoint[1];
      const dz = fish.z - this.startlePoint[2];
      const d = Math.hypot(dx, dy, dz) || 1;
      const power = (this.startleTime / 1.4) * 9;
      if (d < 1.6) {
        ax += (dx / d) * power;
        ay += (dy / d) * power * 0.5;
        az += (dz / d) * power;
        targetSpeed = def.maxSpeed;
      }
    }

    fish.fed = Math.max(0, fish.fed - dt);
    fish.vx += ax * dt;
    fish.vy += ay * dt;
    fish.vz += az * dt;

    // Ease towards the speed this fish wants right now.
    const speed = Math.hypot(fish.vx, fish.vy, fish.vz) || 1e-4;
    const eased = clamp(speed + (targetSpeed - speed) * Math.min(1, dt * 1.8), 0.002, def.maxSpeed);
    const k = eased / speed;
    fish.vx *= k;
    fish.vy *= k;
    fish.vz *= k;

    // Fish rarely swim steeply, and a nearly vertical heading looks wrong.
    const horizontal = Math.hypot(fish.vx, fish.vz);
    const maxClimb = Math.max(horizontal * 0.55, closest ? 0.14 : 0.05);
    fish.vy = clamp(fish.vy, -maxClimb, maxClimb);
  }

  private nearestFood(fish: Fish): Food | null {
    let best: Food | null = null;
    let bestDist = fish.def.bottom ? 0.7 : 1.1;
    for (const f of this.food) {
      if (f.age < 0) continue;
      // Bottom dwellers can only reach what has already sunk.
      if (fish.def.bottom && f.y > 0.34) continue;
      const d = Math.hypot(f.x - fish.x, f.y - fish.y, f.z - fish.z);
      if (d < bestDist) {
        best = f;
        bestDist = d;
      }
    }
    return best;
  }

  private integrate(fish: Fish, dt: number): void {
    fish.x += fish.vx * dt;
    fish.y += fish.vy * dt;
    fish.z += fish.vz * dt;

    const pad = fish.scale * 0.55;
    const limitX = TANK.halfX - pad;
    const limitZ = TANK.halfZ - pad;
    if (Math.abs(fish.x) > limitX) {
      fish.x = Math.sign(fish.x) * limitX;
      fish.vx *= -0.4;
    }
    if (Math.abs(fish.z) > limitZ) {
      fish.z = Math.sign(fish.z) * limitZ;
      fish.vz *= -0.4;
    }
    const floor = sandHeight(fish.x, fish.z) + fish.scale * 0.32;
    if (fish.y < floor) {
      fish.y = floor;
      fish.vy = Math.abs(fish.vy) * 0.3;
    }
    const ceiling = TANK.water - fish.scale * 0.3;
    if (fish.y > ceiling) {
      fish.y = ceiling;
      fish.vy = -Math.abs(fish.vy) * 0.3;
    }

    const speed = Math.hypot(fish.vx, fish.vy, fish.vz);
    const oldX = fish.fx;
    const oldZ = fish.fz;
    if (speed > 0.012) {
      const rate = 1 - Math.exp(-dt * (fish.scale < 0.15 ? 7 : 3.5));
      const fx = fish.fx + (fish.vx / speed - fish.fx) * rate;
      const fy = fish.fy + (fish.vy / speed - fish.fy) * rate;
      const fz = fish.fz + (fish.vz / speed - fish.fz) * rate;
      const l = Math.hypot(fx, fy, fz) || 1;
      fish.fx = fx / l;
      fish.fy = fy / l;
      fish.fz = fz / l;
    }

    // Body bend and bank both come from how quickly the heading is changing.
    const yawRate = (oldZ * fish.fx - oldX * fish.fz) / Math.max(dt, 1e-3);
    fish.turn += (clamp(yawRate * 0.28, -1, 1) - fish.turn) * Math.min(1, dt * 6);

    // Up is world up, tilted into the turn, then made perpendicular to the heading.
    const sideX = -fish.fz;
    const sideZ = fish.fx;
    const roll = fish.turn * 0.4;
    let ux = sideX * roll;
    let uy = 1;
    let uz = sideZ * roll;
    const along = ux * fish.fx + uy * fish.fy + uz * fish.fz;
    ux -= along * fish.fx;
    uy -= along * fish.fy;
    uz -= along * fish.fz;
    const ul = Math.hypot(ux, uy, uz) || 1;
    fish.ux = ux / ul;
    fish.uy = uy / ul;
    fish.uz = uz / ul;

    const beat = fish.def.beatRest + speed * fish.def.beatPerSpeed;
    fish.phase += dt * beat;
  }
}
