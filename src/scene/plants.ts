/** Instanced aquatic plants, all drawn from one bendable leaf-strip mesh. */

import { MeshBuilder } from "../lib/geometry";
import type { Device, Mesh, Pipeline } from "../lib/gpu/types";
import { clamp, mix, TAU } from "../lib/math";
import { createRng, fbm3, type Rng } from "../lib/noise";
import {
  FRAME_GLSL,
  FRAME_WGSL,
  scenePipeline,
  shader,
  shadowPipeline,
  TANK,
  WATER_GLSL,
  WATER_WGSL,
  type DrawContext,
  type Drawable,
  type SceneTargets,
} from "./common";
import { sandHeight, stoneClearance } from "./layout";

const LEAF_VS = `${FRAME_GLSL}
layout(location = 0) in vec3 aPos;
layout(location = 2) in vec2 aUv;
layout(location = 4) in vec4 iBase;     // xyz, height
layout(location = 5) in vec4 iShape;    // yaw, width, phase, broadness
layout(location = 6) in vec2 iBend;     // outward tilt, hue
out vec3 vP; out vec3 vN; out vec2 vUv; out float vHue; out float vBroad;

void main() {
  float t = aUv.y;
  float side = aPos.x;
  float height = iBase.w;
  float yaw = iShape.x, width = iShape.y, phase = iShape.z, broad = iShape.w;
  float tilt = iBend.x;
  vec3 dir = vec3(cos(yaw), 0.0, sin(yaw));
  vec3 across = vec3(-dir.z, 0.0, dir.x);

  float sway = sin(uTime * 0.72 + phase + t * 2.3) + 0.45 * sin(uTime * 1.43 + phase * 1.7 + t * 4.1);
  float bend = t * t * height;
  vec3 spine = vec3(0.0, height * t * (1.0 - tilt * t * 0.42), 0.0);
  spine += dir * (tilt * bend + sway * 0.075 * bend);
  spine += across * (sway * 0.11 * bend + sin(t * 9.0 + phase) * 0.018 * t * (1.0 - broad));

  float narrow = 1.0 - t * 0.82;
  float oval = pow(sin(PI * clamp(t, 0.0, 1.0)), 0.6);
  float w = width * mix(narrow, oval, broad);
  vec3 world = iBase.xyz + spine + across * side * w;
  // A leaf folds slightly along its midrib.
  world.y += abs(side) * w * 0.25 * broad;

  vec3 tangent = normalize(vec3(0.0, height * (1.0 - tilt * t * 0.84), 0.0) + dir * (2.0 * tilt * t * height + 0.1));
  vec3 normal = normalize(cross(across, tangent));
  vN = normalize(normal + across * side * 0.3);
  vP = world;
  vUv = vec2(side, t);
  vHue = iBend.y;
  vBroad = broad;
  gl_Position = uViewProj * vec4(world, 1.0);
}`;

const LEAF_FS = `${FRAME_GLSL}${WATER_GLSL}
in vec3 vP; in vec3 vN; in vec2 vUv; in float vHue; in float vBroad;
out vec4 outColor;

vec3 leafColor(float hue, float t) {
  vec3 base, tip;
  if (hue < 0.34) {
    base = vec3(0.03, 0.17, 0.05);
    tip = vec3(0.26, 0.58, 0.14);
  } else if (hue < 0.67) {
    base = vec3(0.02, 0.2, 0.09);
    tip = vec3(0.16, 0.5, 0.2);
  } else if (hue < 0.9) {
    base = vec3(0.07, 0.2, 0.03);
    tip = vec3(0.48, 0.66, 0.13);
  } else {
    base = vec3(0.3, 0.05, 0.05);
    tip = vec3(0.82, 0.2, 0.12);
  }
  return mix(base, tip, smoothstep(0.0, 0.95, t));
}

void main() {
  vec3 N = normalize(vN);
  vec3 V = normalize(uCamPos - vP);
  if (dot(N, V) < 0.0) N = -N;
  vec3 albedo = leafColor(vHue, vUv.y);
  float rib = smoothstep(0.1, 0.0, abs(vUv.x));
  albedo = mix(albedo, albedo * 1.6 + 0.02, rib * 0.7);
  if (vBroad > 0.5) {
    float veins = abs(sin((vUv.y * 12.0 - abs(vUv.x) * 3.2) * PI));
    albedo *= 0.86 + 0.26 * smoothstep(0.0, 0.2, veins);
  }
  vec3 col = shade(albedo, N, vP, 28.0, 0.1, 0.7);
  outColor = vec4(applyWater(col, vP), 1.0);
}`;

const LEAF_WGSL = `${FRAME_WGSL}${WATER_WGSL}
struct VIn {
  @location(0) pos: vec3f,
  @location(2) uv: vec2f,
  @location(4) base: vec4f,   // xyz, height
  @location(5) shape: vec4f,  // yaw, width, phase, broadness
  @location(6) bendHue: vec2f, // outward tilt, hue
}
struct VOut {
  @builtin(position) clip: vec4f,
  @location(0) p: vec3f,
  @location(1) n: vec3f,
  @location(2) uv: vec2f,
  @location(3) hue: f32,
  @location(4) broad: f32,
}

@vertex fn vs(v: VIn) -> VOut {
  let t = v.uv.y;
  let side = v.pos.x;
  let height = v.base.w;
  let yaw = v.shape.x;
  let width = v.shape.y;
  let phase = v.shape.z;
  let broad = v.shape.w;
  let tilt = v.bendHue.x;
  let dir = vec3f(cos(yaw), 0.0, sin(yaw));
  let across = vec3f(-dir.z, 0.0, dir.x);

  let sway = sin(frame.time * 0.72 + phase + t * 2.3) + 0.45 * sin(frame.time * 1.43 + phase * 1.7 + t * 4.1);
  let bend = t * t * height;
  var spine = vec3f(0.0, height * t * (1.0 - tilt * t * 0.42), 0.0);
  spine += dir * (tilt * bend + sway * 0.075 * bend);
  spine += across * (sway * 0.11 * bend + sin(t * 9.0 + phase) * 0.018 * t * (1.0 - broad));

  let narrow = 1.0 - t * 0.82;
  let oval = pow(sin(PI * clamp(t, 0.0, 1.0)), 0.6);
  let w = width * mix(narrow, oval, broad);
  var world = v.base.xyz + spine + across * side * w;
  // A leaf folds slightly along its midrib.
  world.y += abs(side) * w * 0.25 * broad;

  let tangent = normalize(vec3f(0.0, height * (1.0 - tilt * t * 0.84), 0.0) + dir * (2.0 * tilt * t * height + 0.1));
  let normal = normalize(cross(across, tangent));
  var o: VOut;
  o.n = normalize(normal + across * side * 0.3);
  o.p = world;
  o.uv = vec2f(side, t);
  o.hue = v.bendHue.y;
  o.broad = broad;
  o.clip = project(world);
  return o;
}

fn leafColor(hue: f32, t: f32) -> vec3f {
  var base: vec3f;
  var tip: vec3f;
  if (hue < 0.34) {
    base = vec3f(0.03, 0.17, 0.05);
    tip = vec3f(0.26, 0.58, 0.14);
  } else if (hue < 0.67) {
    base = vec3f(0.02, 0.2, 0.09);
    tip = vec3f(0.16, 0.5, 0.2);
  } else if (hue < 0.9) {
    base = vec3f(0.07, 0.2, 0.03);
    tip = vec3f(0.48, 0.66, 0.13);
  } else {
    base = vec3f(0.3, 0.05, 0.05);
    tip = vec3f(0.82, 0.2, 0.12);
  }
  return mix(base, tip, smoothstep(0.0, 0.95, t));
}

@fragment fn fs(v: VOut) -> @location(0) vec4f {
  var N = normalize(v.n);
  let V = normalize(frame.camPos - v.p);
  if (dot(N, V) < 0.0) { N = -N; }
  var albedo = leafColor(v.hue, v.uv.y);
  let rib = smoothstep(0.1, 0.0, abs(v.uv.x));
  albedo = mix(albedo, albedo * 1.6 + 0.02, rib * 0.7);
  if (v.broad > 0.5) {
    let veins = abs(sin((v.uv.y * 12.0 - abs(v.uv.x) * 3.2) * PI));
    albedo *= 0.86 + 0.26 * smoothstep(0.0, 0.2, veins);
  }
  let col = shade(albedo, N, v.p, 28.0, 0.1, 0.7);
  return vec4f(applyWater(col, v.p), 1.0);
}`;

const FLOATS_PER_INSTANCE = 10;
const INSTANCE_LAYOUT = [4, 4, 2];

function push(
  out: number[],
  x: number,
  z: number,
  height: number,
  yaw: number,
  width: number,
  phase: number,
  broad: number,
  tilt: number,
  hue: number,
): void {
  out.push(x, sandHeight(x, z) - 0.008, z, height, yaw, width, phase, broad, tilt, hue);
}

function plantValley(out: number[], rng: Rng): void {
  // Tall ribbon grass, thick along the back glass and in the corners.
  for (let i = 0; i < 720; i++) {
    const x = mix(-TANK.halfX + 0.03, TANK.halfX - 0.03, rng());
    const z = mix(-TANK.halfZ + 0.03, 0.05, Math.pow(rng(), 1.6));
    const clump = fbm3(x * 3.5, 1, z * 3.5, 2);
    const corner = Math.abs(x) / TANK.halfX;
    const backness = 1 - clamp((z + TANK.halfZ) / 0.7, 0, 1);
    if (clump + backness * 0.3 + corner * 0.15 < 0.62) continue;
    if (stoneClearance(x, z) < 0.25) continue;
    const height = mix(0.42, 1.02, Math.pow(rng(), 0.7)) * (0.72 + 0.3 * backness);
    push(
      out,
      x,
      z,
      height,
      rng() * TAU,
      mix(0.009, 0.017, rng()),
      rng() * TAU,
      0,
      0.1 + rng() * 0.25,
      rng() * 0.66,
    );
  }
}

function plantRosette(
  out: number[],
  rng: Rng,
  x: number,
  z: number,
  leaves: number,
  size: number,
  hue: number,
): void {
  for (let i = 0; i < leaves; i++) {
    const ring = i / leaves;
    const dist = 0.008 + rng() * 0.018;
    const yaw = i * 2.399 + rng() * 0.3;
    push(
      out,
      x + Math.cos(yaw) * dist,
      z + Math.sin(yaw) * dist,
      size * mix(1, 0.58, ring) * (0.9 + rng() * 0.2),
      yaw,
      size * (0.075 + rng() * 0.03),
      rng() * TAU,
      1,
      0.22 + ring * 0.85,
      hue + rng() * 0.12 - 0.06,
    );
  }
}

function plantCarpet(out: number[], rng: Rng): void {
  // Short foreground tufts, leaving a bare sand path through the centre.
  for (let i = 0; i < 1800; i++) {
    const x = mix(-TANK.halfX + 0.02, TANK.halfX - 0.02, rng());
    const z = mix(-0.15, TANK.halfZ - 0.03, rng());
    const patch = fbm3(x * 2.3 + 3, 9, z * 4.2, 3);
    const path = Math.abs(x + 0.08 - z * 0.25);
    if (patch < 0.5 || path < 0.26) continue;
    if (stoneClearance(x, z) < 0.12) continue;
    push(
      out,
      x,
      z,
      mix(0.025, 0.07, rng()),
      rng() * TAU,
      mix(0.006, 0.011, rng()),
      rng() * TAU,
      0.6,
      0.5 + rng() * 0.7,
      0.68 + rng() * 0.2,
    );
  }
}

export class Plants implements Drawable {
  private readonly pipeline: Pipeline;
  private readonly shadow: Pipeline;
  private readonly mesh: Mesh;

  constructor(device: Device, targets: SceneTargets) {
    const source = shader(LEAF_VS, LEAF_FS, LEAF_WGSL);
    const extra = { instances: INSTANCE_LAYOUT };
    this.pipeline = scenePipeline(device, targets, "plants", source, extra);
    this.shadow = shadowPipeline(device, "plants", source, extra);

    const builder = new MeshBuilder();
    builder.grid(2, 16, (s, t) => ({ p: [s * 2 - 1, 0, 0], n: [0, 0, 1], u: s * 2 - 1, v: t }));
    this.mesh = device.createMesh(builder.build(), INSTANCE_LAYOUT);

    const rng = createRng(4242);
    const data: number[] = [];
    plantValley(data, rng);
    plantRosette(data, rng, -0.18, -0.42, 22, 0.62, 0.5);
    plantRosette(data, rng, 1.3, 0.0, 16, 0.5, 0.3);
    plantRosette(data, rng, -1.32, 0.3, 14, 0.42, 0.94);
    plantRosette(data, rng, 0.62, -0.46, 14, 0.46, 0.94);
    plantRosette(data, rng, -0.34, 0.26, 10, 0.2, 0.8);
    plantRosette(data, rng, 0.72, 0.34, 11, 0.2, 0.8);
    plantCarpet(data, rng);
    this.mesh.setInstances(new Float32Array(data), data.length / FLOATS_PER_INSTANCE);
  }

  draw({ pass, frame, water, shadow }: DrawContext): void {
    pass.draw(shadow ? this.shadow : this.pipeline, this.mesh, { frame, textures: water });
  }
}
