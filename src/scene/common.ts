/** Tank dimensions, per-frame data, and the shader code shared by every underwater material. */

import type {
  Device,
  Pass,
  Pipeline,
  PipelineDesc,
  ShaderSource,
  Texture,
  TextureBinding,
  Uniforms,
} from "../lib/gpu/types";
import { normalize, type Floats, type Vec3 } from "../lib/math";

export const TANK = {
  halfX: 1.5,
  halfZ: 0.65,
  height: 1.4,
  water: 1.28,
} as const;

export const WATER_MIN: Vec3 = [-TANK.halfX, 0, -TANK.halfZ];
export const WATER_MAX: Vec3 = [TANK.halfX, TANK.water, TANK.halfZ];
export const LIGHT_DIR: Vec3 = normalize([0.22, 1, 0.1]);

const f = (x: number): string => x.toFixed(6);

/* ---------------------------------------------------------------- GLSL ---------------------- */

/** Available in vertex and fragment shaders. */
export const FRAME_GLSL = `
layout(std140) uniform Frame {
  mat4 uViewProj;
  mat4 uLightVP;
  vec3 uCamPos;
  float uTime;
  vec3 uLightCol;
  float uDay;
  vec3 uScatterCol;
  float uPad;
};
const float WATER_Y = ${f(TANK.water)};
const vec3 WATER_MIN = vec3(${f(-TANK.halfX)}, 0.0, ${f(-TANK.halfZ)});
const vec3 WATER_MAX = vec3(${f(TANK.halfX)}, WATER_Y, ${f(TANK.halfZ)});
const vec3 LIGHT_DIR = vec3(${f(LIGHT_DIR[0])}, ${f(LIGHT_DIR[1])}, ${f(LIGHT_DIR[2])});
const float PI = 3.14159265;
`;

export const NOISE_GLSL = `
float hash13(vec3 p) {
  p = fract(p * 0.1031);
  p += dot(p, p.zyx + 31.32);
  return fract((p.x + p.y) * p.z);
}
float vnoise(vec3 p) {
  vec3 i = floor(p), q = fract(p);
  q = q * q * (3.0 - 2.0 * q);
  return mix(
    mix(mix(hash13(i), hash13(i + vec3(1, 0, 0)), q.x),
        mix(hash13(i + vec3(0, 1, 0)), hash13(i + vec3(1, 1, 0)), q.x), q.y),
    mix(mix(hash13(i + vec3(0, 0, 1)), hash13(i + vec3(1, 0, 1)), q.x),
        mix(hash13(i + vec3(0, 1, 1)), hash13(i + vec3(1, 1, 1)), q.x), q.y),
    q.z);
}
`;

/** Fragment-only: caustics, shadows, Beer-Lambert absorption and in-scatter along the view ray. */
export const WATER_GLSL = `
uniform sampler2D uCaustics;
uniform sampler2DShadow uShadow;

const vec3 ABSORB = vec3(0.42, 0.12, 0.07);
const vec3 FOG = vec3(0.34, 0.15, 0.12);

vec2 boxRay(vec3 ro, vec3 rd) {
  vec3 m = 1.0 / rd;
  vec3 a = (WATER_MIN - ro) * m, b = (WATER_MAX - ro) * m;
  vec3 lo = min(a, b), hi = max(a, b);
  return vec2(max(max(lo.x, lo.y), lo.z), min(min(hi.x, hi.y), hi.z));
}

vec2 lightUv(vec3 P) {
  return (P.xz + LIGHT_DIR.xz / LIGHT_DIR.y * (WATER_Y - P.y)) * 0.62;
}

vec3 causticAt(vec3 P) {
  vec2 uv = lightUv(P);
  float spread = 0.0012 + 0.002 * max(WATER_Y - P.y, 0.0);
  return vec3(
    texture(uCaustics, uv + vec2(spread, 0.0)).r,
    texture(uCaustics, uv).r,
    texture(uCaustics, uv - vec2(spread, 0.0)).r);
}

float shadowAt(vec3 P) {
  vec3 p = (uLightVP * vec4(P, 1.0)).xyz * 0.5 + 0.5;
  p.z -= 0.003;
  const float d = 1.6 / 1024.0;
  return 0.25 * (
    texture(uShadow, p + vec3(-d, -d * 0.4, 0.0)) +
    texture(uShadow, p + vec3(d, d * 0.4, 0.0)) +
    texture(uShadow, p + vec3(-d * 0.4, d, 0.0)) +
    texture(uShadow, p + vec3(d * 0.4, -d, 0.0)));
}

/** Applies water between the camera and P. Works for anything, inside the tank or behind it. */
vec3 applyWater(vec3 col, vec3 P) {
  vec3 d = P - uCamPos;
  float len = length(d);
  vec3 rd = d / len;
  vec2 t = boxRay(uCamPos, rd);
  float a = max(t.x, 0.0), b = min(t.y, len);
  float path = b - a;
  if (path <= 0.0) return col;
  vec3 tr = exp(-path * FOG);
  float y = uCamPos.y + rd.y * (a + b) * 0.5;
  vec3 scatter = uScatterCol * (0.4 + 0.9 * smoothstep(0.0, WATER_Y, y));
  return col * tr + scatter * (1.0 - tr);
}

vec3 shade(vec3 albedo, vec3 N, vec3 P, float gloss, float specular, float translucency) {
  vec3 V = normalize(uCamPos - P);
  vec3 atten = exp(-max(WATER_Y - P.y, 0.0) * ABSORB);
  float sh = shadowAt(P + N * 0.012);
  float ndl = dot(N, LIGHT_DIR);
  float diffuse = max(ndl, 0.0) + translucency * max(-ndl, 0.0);
  // Caustics stretch into streaks on steep surfaces, so they fade out there.
  vec3 caustic = mix(vec3(0.22), causticAt(P), smoothstep(0.2, 0.75, N.y));
  vec3 direct = uLightCol * atten * diffuse * sh * (0.5 + caustic * 1.25);
  float heightFade = 0.45 + 0.55 * smoothstep(0.0, WATER_Y, P.y);
  vec3 ambient = mix(uScatterCol * 0.5, uScatterCol * 1.5 + uLightCol * 0.05, N.y * 0.5 + 0.5) * heightFade;
  vec3 H = normalize(V + LIGHT_DIR);
  vec3 spec = uLightCol * atten * sh * specular * pow(max(dot(N, H), 0.0), gloss);
  return albedo * (direct + ambient) + spec;
}
`;

/** Vertex shader for a full-screen triangle, with vUv running 0 to 1 across the target. */
export const FULLSCREEN_GLSL = `
out vec2 vUv;
void main() {
  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  vUv = p;
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

/* ---------------------------------------------------------------- WGSL ---------------------- */

/**
 * Starts every WGSL module. The scene samples textures inside branches the way GLSL allows, so the
 * uniformity check for implicit derivatives is relaxed to match.
 */
export const PRELUDE_WGSL = `
diagnostic(off, derivative_uniformity);
const PI = 3.14159265;
`;

/**
 * The frame block, at group 0 binding 0. The matrices are built for GL's -1..1 clip depth, so
 * \`toClip\` remaps depth to WebGPU's 0..1; depth values then match what WebGL2 stores.
 */
export const FRAME_WGSL = `${PRELUDE_WGSL}
struct Frame {
  viewProj: mat4x4f,
  lightVP: mat4x4f,
  camPos: vec3f,
  time: f32,
  lightCol: vec3f,
  day: f32,
  scatterCol: vec3f,
  pad: f32,
}
@group(0) @binding(0) var<uniform> frame: Frame;
const WATER_Y = ${f(TANK.water)};
const WATER_MIN = vec3f(${f(-TANK.halfX)}, 0.0, ${f(-TANK.halfZ)});
const WATER_MAX = vec3f(${f(TANK.halfX)}, WATER_Y, ${f(TANK.halfZ)});
const LIGHT_DIR = vec3f(${f(LIGHT_DIR[0])}, ${f(LIGHT_DIR[1])}, ${f(LIGHT_DIR[2])});

fn toClip(p: vec4f) -> vec4f {
  return vec4f(p.xy, (p.z + p.w) * 0.5, p.w);
}
fn project(world: vec3f) -> vec4f {
  return toClip(frame.viewProj * vec4f(world, 1.0));
}
`;

export const NOISE_WGSL = `
fn hash13(p0: vec3f) -> f32 {
  var p = fract(p0 * 0.1031);
  p += dot(p, p.zyx + 31.32);
  return fract((p.x + p.y) * p.z);
}
fn vnoise(p: vec3f) -> f32 {
  let i = floor(p);
  var q = fract(p);
  q = q * q * (3.0 - 2.0 * q);
  return mix(
    mix(mix(hash13(i), hash13(i + vec3f(1, 0, 0)), q.x),
        mix(hash13(i + vec3f(0, 1, 0)), hash13(i + vec3f(1, 1, 0)), q.x), q.y),
    mix(mix(hash13(i + vec3f(0, 0, 1)), hash13(i + vec3f(1, 0, 1)), q.x),
        mix(hash13(i + vec3f(0, 1, 1)), hash13(i + vec3f(1, 1, 1)), q.x), q.y),
    q.z);
}
`;

/** WGSL twin of WATER_GLSL. Textures sit at bindings 1 to 4, matching WATER_TEXTURES. */
export const WATER_WGSL = `
@group(0) @binding(1) var causticsTex: texture_2d<f32>;
@group(0) @binding(2) var causticsSamp: sampler;
@group(0) @binding(3) var shadowTex: texture_depth_2d;
@group(0) @binding(4) var shadowSamp: sampler_comparison;

const ABSORB = vec3f(0.42, 0.12, 0.07);
const FOG = vec3f(0.34, 0.15, 0.12);

fn boxRay(ro: vec3f, rd: vec3f) -> vec2f {
  let m = 1.0 / rd;
  let a = (WATER_MIN - ro) * m;
  let b = (WATER_MAX - ro) * m;
  let lo = min(a, b);
  let hi = max(a, b);
  return vec2f(max(max(lo.x, lo.y), lo.z), min(min(hi.x, hi.y), hi.z));
}

fn lightUv(P: vec3f) -> vec2f {
  return (P.xz + LIGHT_DIR.xz / LIGHT_DIR.y * (WATER_Y - P.y)) * 0.62;
}

fn causticAt(P: vec3f) -> vec3f {
  let uv = lightUv(P);
  let spread = 0.0012 + 0.002 * max(WATER_Y - P.y, 0.0);
  return vec3f(
    textureSample(causticsTex, causticsSamp, uv + vec2f(spread, 0.0)).r,
    textureSample(causticsTex, causticsSamp, uv).r,
    textureSample(causticsTex, causticsSamp, uv - vec2f(spread, 0.0)).r);
}

/** Shadow-map coordinates. Texture rows run top-down in WebGPU, so y flips. */
fn shadowCoord(P: vec3f) -> vec3f {
  let c = (frame.lightVP * vec4f(P, 1.0)).xyz;
  return vec3f(c.x * 0.5 + 0.5, 0.5 - c.y * 0.5, c.z * 0.5 + 0.5);
}

fn shadowAt(P: vec3f) -> f32 {
  let p = shadowCoord(P);
  let z = p.z - 0.003;
  let d = 1.6 / 1024.0;
  return 0.25 * (
    textureSampleCompareLevel(shadowTex, shadowSamp, p.xy + vec2f(-d, d * 0.4), z) +
    textureSampleCompareLevel(shadowTex, shadowSamp, p.xy + vec2f(d, -d * 0.4), z) +
    textureSampleCompareLevel(shadowTex, shadowSamp, p.xy + vec2f(-d * 0.4, -d), z) +
    textureSampleCompareLevel(shadowTex, shadowSamp, p.xy + vec2f(d * 0.4, d), z));
}

fn applyWater(col: vec3f, P: vec3f) -> vec3f {
  let d = P - frame.camPos;
  let len = length(d);
  let rd = d / len;
  let t = boxRay(frame.camPos, rd);
  let a = max(t.x, 0.0);
  let b = min(t.y, len);
  let path = b - a;
  if (path <= 0.0) { return col; }
  let tr = exp(-path * FOG);
  let y = frame.camPos.y + rd.y * (a + b) * 0.5;
  let scatter = frame.scatterCol * (0.4 + 0.9 * smoothstep(0.0, WATER_Y, y));
  return col * tr + scatter * (1.0 - tr);
}

fn shade(albedo: vec3f, N: vec3f, P: vec3f, gloss: f32, specular: f32, translucency: f32) -> vec3f {
  let V = normalize(frame.camPos - P);
  let atten = exp(-max(WATER_Y - P.y, 0.0) * ABSORB);
  let sh = shadowAt(P + N * 0.012);
  let ndl = dot(N, LIGHT_DIR);
  let diffuse = max(ndl, 0.0) + translucency * max(-ndl, 0.0);
  let caustic = mix(vec3f(0.22), causticAt(P), smoothstep(0.2, 0.75, N.y));
  let direct = frame.lightCol * atten * diffuse * sh * (0.5 + caustic * 1.25);
  let heightFade = 0.45 + 0.55 * smoothstep(0.0, WATER_Y, P.y);
  let ambient = mix(frame.scatterCol * 0.5, frame.scatterCol * 1.5 + frame.lightCol * 0.05, N.y * 0.5 + 0.5) * heightFade;
  let H = normalize(V + LIGHT_DIR);
  let spec = frame.lightCol * atten * sh * specular * pow(max(dot(N, H), 0.0), gloss);
  return albedo * (direct + ambient) + spec;
}
`;

/** WGSL full-screen triangle. \`uv\` is in texture space, which in WebGPU runs top-down. */
export const FULLSCREEN_WGSL = `
struct FsOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f }
@vertex fn vs(@builtin(vertex_index) i: u32) -> FsOut {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  var o: FsOut;
  o.pos = vec4f(p * 2.0 - 1.0, 0.0, 1.0);
  o.uv = vec2f(p.x, 1.0 - p.y);
  return o;
}
`;

/* ---------------------------------------------------------------- Pipelines ----------------- */

/** The two inputs every underwater material samples, in the order their bindings expect. */
export const WATER_TEXTURES: readonly TextureBinding[] = [
  { name: "uCaustics", kind: "float" },
  { name: "uShadow", kind: "shadow" },
];

export function shader(vertex: string, fragment: string, wgsl: string): ShaderSource {
  return { glsl: { vertex, fragment }, wgsl };
}

/** Settings shared by every pipeline that draws into the HDR scene. */
export interface SceneTargets {
  color: "rgba16f" | "rgba8";
  samples: number;
}

type Extra = Omit<Partial<PipelineDesc>, "label" | "shader">;

/** An opaque or blended material in the main pass, lit through the water. */
export function scenePipeline(
  device: Device,
  targets: SceneTargets,
  label: string,
  source: ShaderSource,
  extra: Extra = {},
): Pipeline {
  return device.createPipeline({
    label,
    shader: source,
    vertex: "mesh",
    frame: true,
    textures: WATER_TEXTURES,
    color: targets.color,
    samples: targets.samples,
    depth: "write",
    ...extra,
  });
}

/** The same vertex shader, writing only depth into the shadow map. */
export function shadowPipeline(
  device: Device,
  label: string,
  source: ShaderSource,
  extra: Extra = {},
): Pipeline {
  return device.createPipeline({
    label: `${label} shadow`,
    shader: source,
    vertex: "mesh",
    frame: true,
    depth: "write",
    ...extra,
  });
}

/* ---------------------------------------------------------------- Frame data ---------------- */

/** Per-frame uniforms, mirrored by the Frame block above. */
export class FrameData {
  private readonly data: Floats = new Float32Array(44);

  set(lightVP: Floats, cam: Vec3, time: number, light: Vec3, day: number, scatter: Vec3): void {
    const { data } = this;
    data.set(lightVP, 16);
    data.set(cam, 32);
    data[35] = time;
    data.set(light, 36);
    data[39] = day;
    data.set(scatter, 40);
  }

  /** The shadow pass reuses the same shaders, so each pass gets a copy with its own view. */
  upload(device: Device, viewProj: Floats): Uniforms {
    this.data.set(viewProj, 0);
    return device.uniforms(this.data);
  }
}

/** What a scene object needs to draw itself into a pass. */
export interface DrawContext {
  device: Device;
  pass: Pass;
  frame: Uniforms;
  /** The caustics texture and the shadow map, or none in the shadow pass. */
  water: readonly Texture[];
  shadow: boolean;
}

/** Every scene object draws into the main pass, and casters also into the shadow pass. */
export interface Drawable {
  draw(context: DrawContext): void;
}
