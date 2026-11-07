/** The substrate: a height field, plus a skirt showing the sand's cross-section through the glass. */

import { MeshBuilder } from "../lib/geometry";
import type { Device, Mesh, Pipeline } from "../lib/gpu/types";
import { mix, normalize } from "../lib/math";
import {
  FRAME_GLSL,
  FRAME_WGSL,
  NOISE_GLSL,
  NOISE_WGSL,
  scenePipeline,
  shader,
  TANK,
  WATER_GLSL,
  WATER_WGSL,
  type DrawContext,
  type Drawable,
  type SceneTargets,
} from "./common";
import { sandHeight, stoneClearance } from "./layout";

const VS = `${FRAME_GLSL}
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 3) in float aAo;
out vec3 vP; out vec3 vN; out float vAo;
void main() {
  vP = aPos; vN = aNormal; vAo = aAo;
  gl_Position = uViewProj * vec4(aPos, 1.0);
}`;

const FS = `${FRAME_GLSL}${NOISE_GLSL}${WATER_GLSL}
in vec3 vP; in vec3 vN; in float vAo;
out vec4 outColor;
void main() {
  vec3 P = vP;
  vec3 N = normalize(vN);
  bool skirt = N.y < 0.2;
  float broad = vnoise(P * 5.0);
  float grain = hash13(floor(P * 340.0));
  vec3 albedo = mix(vec3(0.44, 0.35, 0.24), vec3(0.62, 0.52, 0.37), broad);
  albedo *= 0.88 + 0.22 * grain;
  if (grain > 0.965) albedo *= 0.42;
  else if (grain < 0.03) albedo = mix(albedo, vec3(0.86, 0.5, 0.28), 0.6);
  if (skirt) {
    float strata = vnoise(vec3(P.x * 7.0, P.y * 46.0, P.z * 7.0));
    albedo *= 0.36 + 0.26 * strata;
    N = normalize(N + vec3(0.0, 0.5, 0.0));
  } else {
    float e = 0.02;
    float ripples = vnoise(vec3(P.x * 16.0, 0.0, P.z * 42.0 + sin(P.x * 3.0)));
    float rx = vnoise(vec3((P.x + e) * 16.0, 0.0, P.z * 42.0 + sin(P.x * 3.0))) - ripples;
    float rz = vnoise(vec3(P.x * 16.0, 0.0, (P.z + e) * 42.0 + sin(P.x * 3.0))) - ripples;
    N = normalize(N - vec3(rx, 0.0, rz) * 0.55);
  }
  vec3 col = shade(albedo * vAo, N, P, 12.0, 0.015, 0.0);
  outColor = vec4(applyWater(col, P), 1.0);
}`;

const WGSL = `${FRAME_WGSL}${NOISE_WGSL}${WATER_WGSL}
struct VIn { @location(0) pos: vec3f, @location(1) normal: vec3f, @location(3) ao: f32 }
struct VOut {
  @builtin(position) clip: vec4f,
  @location(0) p: vec3f,
  @location(1) n: vec3f,
  @location(2) ao: f32,
}

@vertex fn vs(v: VIn) -> VOut {
  var o: VOut;
  o.p = v.pos;
  o.n = v.normal;
  o.ao = v.ao;
  o.clip = project(v.pos);
  return o;
}

@fragment fn fs(v: VOut) -> @location(0) vec4f {
  let P = v.p;
  var N = normalize(v.n);
  let skirt = N.y < 0.2;
  let broad = vnoise(P * 5.0);
  let grain = hash13(floor(P * 340.0));
  var albedo = mix(vec3f(0.44, 0.35, 0.24), vec3f(0.62, 0.52, 0.37), broad);
  albedo *= 0.88 + 0.22 * grain;
  if (grain > 0.965) {
    albedo *= 0.42;
  } else if (grain < 0.03) {
    albedo = mix(albedo, vec3f(0.86, 0.5, 0.28), 0.6);
  }
  if (skirt) {
    let strata = vnoise(vec3f(P.x * 7.0, P.y * 46.0, P.z * 7.0));
    albedo *= 0.36 + 0.26 * strata;
    N = normalize(N + vec3f(0.0, 0.5, 0.0));
  } else {
    let e = 0.02;
    let ripples = vnoise(vec3f(P.x * 16.0, 0.0, P.z * 42.0 + sin(P.x * 3.0)));
    let rx = vnoise(vec3f((P.x + e) * 16.0, 0.0, P.z * 42.0 + sin(P.x * 3.0))) - ripples;
    let rz = vnoise(vec3f(P.x * 16.0, 0.0, (P.z + e) * 42.0 + sin(P.x * 3.0))) - ripples;
    N = normalize(N - vec3f(rx, 0.0, rz) * 0.55);
  }
  let col = shade(albedo * v.ao, N, P, 12.0, 0.015, 0.0);
  return vec4f(applyWater(col, P), 1.0);
}`;

export class Sand implements Drawable {
  private readonly pipeline: Pipeline;
  private readonly mesh: Mesh;

  constructor(device: Device, targets: SceneTargets) {
    this.pipeline = scenePipeline(device, targets, "sand", shader(VS, FS, WGSL));

    const builder = new MeshBuilder();
    const inset = 0.002;
    const x0 = -TANK.halfX + inset;
    const x1 = TANK.halfX - inset;
    const z0 = -TANK.halfZ + inset;
    const z1 = TANK.halfZ - inset;
    const e = 0.015;
    builder.grid(180, 80, (s, t) => {
      const x = mix(x0, x1, s);
      const z = mix(z0, z1, t);
      const n = normalize([
        sandHeight(x - e, z) - sandHeight(x + e, z),
        2 * e,
        sandHeight(x, z - e) - sandHeight(x, z + e),
      ]);
      const ao = 0.42 + 0.58 * Math.pow(stoneClearance(x, z), 0.8);
      return { p: [x, sandHeight(x, z), z], n, extra: ao };
    });

    // The vertical faces of the sand, seen through the glass.
    const skirt = (
      x: (s: number) => number,
      z: (s: number) => number,
      nx: number,
      nz: number,
    ): void => {
      builder.grid(90, 1, (s, t) => {
        const px = x(s);
        const pz = z(s);
        return { p: [px, t * sandHeight(px, pz), pz], n: [nx, 0, nz], extra: 0.85 };
      });
    };
    skirt(
      (s) => mix(x0, x1, s),
      () => z1,
      0,
      1,
    );
    skirt(
      () => x0,
      (s) => mix(z0, z1, s),
      -1,
      0,
    );
    skirt(
      () => x1,
      (s) => mix(z0, z1, s),
      1,
      0,
    );
    this.mesh = device.createMesh(builder.build());
  }

  draw({ pass, frame, water, shadow }: DrawContext): void {
    if (shadow) return;
    pass.draw(this.pipeline, this.mesh, { frame, textures: water });
  }
}
