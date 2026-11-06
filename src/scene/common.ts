/** Tank dimensions, per-frame data, and the GLSL shared by every underwater material. */

import { FRAME_BINDING } from "../lib/gl";
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

/** Texture units reserved for the two inputs every underwater material samples. */
export const UNIT_CAUSTICS = 0;
export const UNIT_SHADOW = 1;

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

/** Per-frame uniform buffer, mirrored by the Frame block above. */
export class FrameData {
  private readonly gl: WebGL2RenderingContext;
  private readonly buffer: WebGLBuffer;
  private readonly data: Floats = new Float32Array(44);

  constructor(gl: WebGL2RenderingContext) {
    this.gl = gl;
    this.buffer = gl.createBuffer();
    gl.bindBuffer(gl.UNIFORM_BUFFER, this.buffer);
    gl.bufferData(gl.UNIFORM_BUFFER, this.data.byteLength, gl.DYNAMIC_DRAW);
    gl.bindBufferBase(gl.UNIFORM_BUFFER, FRAME_BINDING, this.buffer);
  }

  set(
    viewProj: Floats,
    lightVP: Floats,
    cam: Vec3,
    time: number,
    light: Vec3,
    day: number,
    scatter: Vec3,
  ): void {
    const { data } = this;
    data.set(viewProj, 0);
    data.set(lightVP, 16);
    data.set(cam, 32);
    data[35] = time;
    data.set(light, 36);
    data[39] = day;
    data.set(scatter, 40);
    this.upload();
  }

  /** The shadow pass reuses the same shaders, so it just swaps the view-projection. */
  setViewProj(viewProj: Floats): void {
    this.data.set(viewProj, 0);
    this.upload();
  }

  private upload(): void {
    const { gl } = this;
    gl.bindBuffer(gl.UNIFORM_BUFFER, this.buffer);
    gl.bufferSubData(gl.UNIFORM_BUFFER, 0, this.data);
  }
}

/** Every scene object draws into the main pass, and casters also into the shadow pass. */
export interface Drawable {
  draw(shadowPass: boolean): void;
}

export const SHADOW_FS = `void main() {}`;

/** Binds the samplers for a material; the textures themselves stay bound for the whole frame. */
export function bindWaterSamplers(program: {
  use(): { int(name: string, value: number): unknown };
}): void {
  const p = program.use();
  p.int("uCaustics", UNIT_CAUSTICS);
  p.int("uShadow", UNIT_SHADOW);
}
