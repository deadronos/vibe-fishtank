/** The transparent shell of the tank: the moving water surface and the glass walls. */

import { MeshBuilder } from "../lib/geometry";
import type { Device, Mesh, Pipeline } from "../lib/gpu/types";
import { mix } from "../lib/math";
import {
  FRAME_GLSL,
  FRAME_WGSL,
  scenePipeline,
  shader,
  TANK,
  WATER_GLSL,
  WATER_WGSL,
  type DrawContext,
  type Drawable,
  type SceneTargets,
} from "./common";

const SURFACE_VS = `${FRAME_GLSL}
layout(location = 0) in vec3 aPos;
out vec3 vP; out vec3 vN;

// Each wave: direction xy, frequency, speed. Amplitudes fall with frequency.
const vec4 WAVES[5] = vec4[5](
  vec4( 0.8,  0.6,  7.0, 0.9),
  vec4(-0.5,  0.86, 11.0, 1.3),
  vec4( 0.2, -0.98, 17.0, 1.1),
  vec4(-0.9, -0.42, 27.0, 1.8),
  vec4( 0.63, 0.77, 41.0, 2.3));

void main() {
  vec3 p = aPos;
  float h = 0.0;
  vec2 g = vec2(0.0);
  for (int i = 0; i < 5; i++) {
    vec4 w = WAVES[i];
    float a = 0.018 / w.z * 3.2;
    float s = dot(w.xy, p.xz) * w.z + uTime * w.w;
    h += sin(s) * a;
    g += w.xy * cos(s) * a * w.z;
  }
  // Pin the edges against the glass.
  vec2 edge = min(WATER_MAX.xz - abs(p.xz), vec2(0.06)) / 0.06;
  float pin = edge.x * edge.y;
  p.y += h * pin;
  vP = p;
  vN = normalize(vec3(-g.x * pin, 1.0, -g.y * pin));
  gl_Position = uViewProj * vec4(p, 1.0);
}`;

const SURFACE_FS = `${FRAME_GLSL}${WATER_GLSL}
in vec3 vP; in vec3 vN;
out vec4 outColor;
void main() {
  vec3 N = normalize(vN);
  vec3 V = normalize(uCamPos - vP);
  vec3 col;
  float alpha;
  if (uCamPos.y < vP.y) {
    // From underneath: light comes through Snell's window, the rest mirrors the water below.
    vec3 n = -N;
    float cosI = max(dot(V, n), 0.0);
    float sinT = sqrt(max(1.0 - cosI * cosI, 0.0)) * 1.33;
    float window = 1.0 - smoothstep(0.82, 1.0, sinT);
    float shimmer = 0.6 + 2.0 * texture(uCaustics, vP.xz * 0.9).r;
    vec3 mirrored = uScatterCol * (1.1 + 0.8 * shimmer) + uLightCol * 0.035 * shimmer;
    vec3 through = uLightCol * (0.5 + 0.35 * shimmer);
    col = mix(mirrored, through, window);
    alpha = mix(0.86, 0.75, window);
  } else {
    float cosI = max(dot(V, N), 0.0);
    float fresnel = 0.02 + 0.98 * pow(1.0 - cosI, 5.0);
    vec3 R = reflect(-V, N);
    vec3 env = uLightCol * 0.5 * smoothstep(0.75, 1.0, R.y) + vec3(0.02, 0.03, 0.035);
    col = env + uScatterCol * 0.25;
    alpha = 0.25 + fresnel * 0.7;
  }
  outColor = vec4(applyWater(col, vP), alpha);
}`;

const GLASS_VS = `${FRAME_GLSL}
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aUv;
out vec3 vP; out vec3 vN; out vec2 vUv;
void main() {
  vP = aPos; vN = aNormal; vUv = aUv;
  gl_Position = uViewProj * vec4(aPos, 1.0);
}`;

const GLASS_FS = `${FRAME_GLSL}
in vec3 vP; in vec3 vN; in vec2 vUv;
out vec4 outColor;
void main() {
  vec3 N = normalize(vN);
  vec3 V = normalize(uCamPos - vP);
  if (dot(N, V) < 0.0) N = -N;
  float fresnel = 0.03 + 0.97 * pow(1.0 - max(dot(N, V), 0.0), 5.0);
  vec3 R = reflect(-V, N);

  // A dim room to reflect, with a large soft window off to the side.
  float window = smoothstep(0.1, 0.5, R.y) * smoothstep(0.75, 0.2, R.y)
    * smoothstep(-0.7, -0.35, R.x) * smoothstep(0.05, -0.25, R.x);
  vec3 env = mix(vec3(0.012, 0.015, 0.02), vec3(0.07, 0.08, 0.1), R.y * 0.5 + 0.5);
  env += vec3(0.75, 0.82, 0.92) * window * mix(0.08, 0.3, uDay);

  vec3 col = env * fresnel;
  float alpha = clamp(fresnel * 1.2, 0.0, 0.42) + 0.012;

  // Light travels inside the glass and escapes at the polished edges.
  vec2 d = min(vUv, 1.0 - vUv) * vec2(3.0, 1.4);
  float edge = smoothstep(0.012, 0.0, min(d.x, d.y));
  vec3 lit = uLightCol * 0.06 + vec3(0.015, 0.04, 0.035);
  col += lit * edge;
  alpha = max(alpha, edge * 0.4);

  // The meniscus, where the water climbs the glass.
  float meniscus = smoothstep(0.009, 0.0, abs(vP.y - WATER_Y - 0.003));
  col += (uLightCol * 0.14 + uScatterCol) * meniscus;
  alpha = max(alpha, meniscus * 0.7);

  outColor = vec4(col, alpha);
}`;

const SURFACE_WGSL = `${FRAME_WGSL}${WATER_WGSL}
struct VOut { @builtin(position) clip: vec4f, @location(0) p: vec3f, @location(1) n: vec3f }

@vertex fn vs(@location(0) pos: vec3f) -> VOut {
  // Each wave: direction xy, frequency, speed. Amplitudes fall with frequency.
  var waves = array<vec4f, 5>(
    vec4f( 0.8,  0.6,  7.0, 0.9),
    vec4f(-0.5,  0.86, 11.0, 1.3),
    vec4f( 0.2, -0.98, 17.0, 1.1),
    vec4f(-0.9, -0.42, 27.0, 1.8),
    vec4f( 0.63, 0.77, 41.0, 2.3));
  var p = pos;
  var h = 0.0;
  var g = vec2f(0.0);
  for (var i = 0; i < 5; i++) {
    let w = waves[i];
    let a = 0.018 / w.z * 3.2;
    let s = dot(w.xy, p.xz) * w.z + frame.time * w.w;
    h += sin(s) * a;
    g += w.xy * cos(s) * a * w.z;
  }
  // Pin the edges against the glass.
  let edge = min(WATER_MAX.xz - abs(p.xz), vec2f(0.06)) / 0.06;
  let pin = edge.x * edge.y;
  p.y += h * pin;
  var o: VOut;
  o.p = p;
  o.n = normalize(vec3f(-g.x * pin, 1.0, -g.y * pin));
  o.clip = project(p);
  return o;
}

@fragment fn fs(v: VOut) -> @location(0) vec4f {
  let N = normalize(v.n);
  let V = normalize(frame.camPos - v.p);
  var col: vec3f;
  var alpha: f32;
  if (frame.camPos.y < v.p.y) {
    // From underneath: light comes through Snell's window, the rest mirrors the water below.
    let n = -N;
    let cosI = max(dot(V, n), 0.0);
    let sinT = sqrt(max(1.0 - cosI * cosI, 0.0)) * 1.33;
    let window = 1.0 - smoothstep(0.82, 1.0, sinT);
    let shimmer = 0.6 + 2.0 * textureSample(causticsTex, causticsSamp, v.p.xz * 0.9).r;
    let mirrored = frame.scatterCol * (1.1 + 0.8 * shimmer) + frame.lightCol * 0.035 * shimmer;
    let through = frame.lightCol * (0.5 + 0.35 * shimmer);
    col = mix(mirrored, through, window);
    alpha = mix(0.86, 0.75, window);
  } else {
    let cosI = max(dot(V, N), 0.0);
    let fresnel = 0.02 + 0.98 * pow(1.0 - cosI, 5.0);
    let R = reflect(-V, N);
    let env = frame.lightCol * 0.5 * smoothstep(0.75, 1.0, R.y) + vec3f(0.02, 0.03, 0.035);
    col = env + frame.scatterCol * 0.25;
    alpha = 0.25 + fresnel * 0.7;
  }
  return vec4f(applyWater(col, v.p), alpha);
}`;

const GLASS_WGSL = `${FRAME_WGSL}
struct VIn { @location(0) pos: vec3f, @location(1) normal: vec3f, @location(2) uv: vec2f }
struct VOut {
  @builtin(position) clip: vec4f,
  @location(0) p: vec3f,
  @location(1) n: vec3f,
  @location(2) uv: vec2f,
}

@vertex fn vs(v: VIn) -> VOut {
  var o: VOut;
  o.p = v.pos;
  o.n = v.normal;
  o.uv = v.uv;
  o.clip = project(v.pos);
  return o;
}

@fragment fn fs(v: VOut) -> @location(0) vec4f {
  var N = normalize(v.n);
  let V = normalize(frame.camPos - v.p);
  if (dot(N, V) < 0.0) { N = -N; }
  let fresnel = 0.03 + 0.97 * pow(1.0 - max(dot(N, V), 0.0), 5.0);
  let R = reflect(-V, N);

  // A dim room to reflect, with a large soft window off to the side.
  let window = smoothstep(0.1, 0.5, R.y) * smoothstep(0.75, 0.2, R.y)
    * smoothstep(-0.7, -0.35, R.x) * smoothstep(0.05, -0.25, R.x);
  var env = mix(vec3f(0.012, 0.015, 0.02), vec3f(0.07, 0.08, 0.1), R.y * 0.5 + 0.5);
  env += vec3f(0.75, 0.82, 0.92) * window * mix(0.08, 0.3, frame.day);

  var col = env * fresnel;
  var alpha = clamp(fresnel * 1.2, 0.0, 0.42) + 0.012;

  // Light travels inside the glass and escapes at the polished edges.
  let d = min(v.uv, 1.0 - v.uv) * vec2f(3.0, 1.4);
  let edge = smoothstep(0.012, 0.0, min(d.x, d.y));
  let lit = frame.lightCol * 0.06 + vec3f(0.015, 0.04, 0.035);
  col += lit * edge;
  alpha = max(alpha, edge * 0.4);

  // The meniscus, where the water climbs the glass.
  let meniscus = smoothstep(0.009, 0.0, abs(v.p.y - WATER_Y - 0.003));
  col += (frame.lightCol * 0.14 + frame.scatterCol) * meniscus;
  alpha = max(alpha, meniscus * 0.7);

  return vec4f(col, alpha);
}`;

/** Transparent layers test depth but don't write it, and blend over what's behind them. */
const TRANSPARENT = { depth: "test", blend: "alpha" } as const;

export class WaterSurface implements Drawable {
  private readonly pipeline: Pipeline;
  private readonly mesh: Mesh;

  constructor(device: Device, targets: SceneTargets) {
    const source = shader(SURFACE_VS, SURFACE_FS, SURFACE_WGSL);
    this.pipeline = scenePipeline(device, targets, "water surface", source, TRANSPARENT);
    const builder = new MeshBuilder();
    builder.grid(180, 80, (s, t) => ({
      p: [mix(-TANK.halfX, TANK.halfX, s), TANK.water, mix(-TANK.halfZ, TANK.halfZ, t)],
      n: [0, 1, 0],
    }));
    this.mesh = device.createMesh(builder.build());
  }

  draw({ pass, frame, water, shadow }: DrawContext): void {
    if (shadow) return;
    pass.draw(this.pipeline, this.mesh, { frame, textures: water });
  }
}

export class Glass implements Drawable {
  private readonly pipeline: Pipeline;
  private readonly mesh: Mesh;

  constructor(device: Device, targets: SceneTargets) {
    const source = shader(GLASS_VS, GLASS_FS, GLASS_WGSL);
    this.pipeline = scenePipeline(device, targets, "glass", source, {
      ...TRANSPARENT,
      textures: [],
    });
    const { halfX: x, halfZ: z, height: h } = TANK;
    const builder = new MeshBuilder();
    builder.quad([-x, 0, -z], [-x, 0, z], [-x, h, z], [-x, h, -z], [-1, 0, 0]);
    builder.quad([x, 0, z], [x, 0, -z], [x, h, -z], [x, h, z], [1, 0, 0]);
    builder.quad([-x, 0, z], [x, 0, z], [x, h, z], [-x, h, z], [0, 0, 1]);
    this.mesh = device.createMesh(builder.build());
  }

  draw({ pass, frame, shadow }: DrawContext): void {
    if (shadow) return;
    pass.draw(this.pipeline, this.mesh, { frame });
  }
}
