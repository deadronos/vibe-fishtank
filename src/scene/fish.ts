/** Procedural fish: lofted bodies, fin ribbons, swimming in the vertex shader, patterns in the fragment. */

import { MeshBuilder } from "../lib/geometry";
import type { Device, Mesh, Pipeline } from "../lib/gpu/types";
import { mix, normalize, type Floats, type Vec3 } from "../lib/math";
import { SPECIES, type Fish } from "../sim/school";
import {
  FRAME_GLSL,
  FRAME_WGSL,
  NOISE_GLSL,
  NOISE_WGSL,
  scenePipeline,
  shader,
  shadowPipeline,
  WATER_GLSL,
  WATER_WGSL,
  type DrawContext,
  type Drawable,
  type SceneTargets,
} from "./common";

/** Fish meshes are one unit long, nose at x = 0.5 and tail tip at x = -0.5, swimming along +x. */
interface Fin {
  from: number;
  to: number;
  height: number;
  sweep: number;
  /** Rises to a long point near the front, rather than a rounded edge. */
  pointed?: boolean;
}

interface BodyPlan {
  halfHeight: number;
  halfWidth: number;
  /** Bigger pushes the deepest part of the body towards the nose. */
  bluntness: number;
  /** Fullness of the body outline. */
  fullness: number;
  bodyEnd: number;
  tail: { rootHeight: number; spread: number; fork: number };
  dorsal: Fin;
  anal: Fin;
  pectoral: number;
  /** Eye position in mesh space, and radius. */
  eye: Vec3;
  wave: number;
  amplitude: number;
}

const PLANS: readonly BodyPlan[] = [
  // Neon tetra
  {
    halfHeight: 0.14,
    halfWidth: 0.055,
    bluntness: 0.72,
    fullness: 0.75,
    bodyEnd: -0.26,
    tail: { rootHeight: 0.03, spread: 0.15, fork: 0.1 },
    dorsal: { from: 0.1, to: -0.02, height: 0.1, sweep: 0.05 },
    anal: { from: 0.0, to: -0.2, height: 0.07, sweep: 0.04 },
    pectoral: 0.09,
    eye: [0.355, 0.03, 0.03],
    wave: 7,
    amplitude: 0.1,
  },
  // Ember rasbora
  {
    halfHeight: 0.15,
    halfWidth: 0.06,
    bluntness: 0.68,
    fullness: 0.7,
    bodyEnd: -0.27,
    tail: { rootHeight: 0.03, spread: 0.16, fork: 0.12 },
    dorsal: { from: 0.04, to: -0.08, height: 0.09, sweep: 0.05 },
    anal: { from: -0.02, to: -0.17, height: 0.07, sweep: 0.03 },
    pectoral: 0.08,
    eye: [0.36, 0.035, 0.034],
    wave: 7,
    amplitude: 0.11,
  },
  // Angelfish: deep body with long swept fins
  {
    halfHeight: 0.33,
    halfWidth: 0.05,
    bluntness: 0.85,
    fullness: 0.62,
    bodyEnd: -0.17,
    tail: { rootHeight: 0.05, spread: 0.2, fork: -0.04 },
    dorsal: { from: 0.14, to: -0.15, height: 0.36, sweep: 0.52, pointed: true },
    anal: { from: 0.1, to: -0.15, height: 0.4, sweep: 0.56, pointed: true },
    pectoral: 0.08,
    eye: [0.3, 0.05, 0.022],
    wave: 5,
    amplitude: 0.05,
  },
  // Discus: nearly round
  {
    halfHeight: 0.36,
    halfWidth: 0.065,
    bluntness: 0.95,
    fullness: 0.5,
    bodyEnd: -0.3,
    tail: { rootHeight: 0.05, spread: 0.13, fork: -0.02 },
    dorsal: { from: 0.2, to: -0.3, height: 0.07, sweep: 0.03 },
    anal: { from: 0.1, to: -0.3, height: 0.07, sweep: 0.03 },
    pectoral: 0.08,
    eye: [0.3, 0.07, 0.03],
    wave: 5,
    amplitude: 0.05,
  },
  // Corydoras: stout and low
  {
    halfHeight: 0.15,
    halfWidth: 0.095,
    bluntness: 0.82,
    fullness: 0.56,
    bodyEnd: -0.27,
    tail: { rootHeight: 0.035, spread: 0.13, fork: 0.07 },
    dorsal: { from: 0.18, to: 0.02, height: 0.14, sweep: 0.06 },
    anal: { from: -0.05, to: -0.16, height: 0.06, sweep: 0.03 },
    pectoral: 0.11,
    eye: [0.34, 0.05, 0.03],
    wave: 7,
    amplitude: 0.08,
  },
];

/** The outline of a dorsal or anal fin, from its leading edge to its trailing edge. */
const finShape = (s: number, pointed: boolean): number => {
  if (pointed)
    return s < 0.22
      ? Math.pow(s / 0.22, 1.3)
      : Math.pow((1 - s) / 0.78, 1.5) * 0.92 + 0.08 * (1 - s);
  return Math.pow(Math.sin(Math.PI * Math.min(1, s * 0.82 + 0.1)), 0.7);
};

function buildFish(plan: BodyPlan): MeshBuilder {
  const b = new MeshBuilder();
  const NOSE = 0.5;
  const bodyLength = NOSE - plan.bodyEnd;

  const outline = (x: number): number => {
    const s = Math.min(1, Math.max(0, (NOSE - x) / bodyLength));
    const body = Math.pow(Math.sin(Math.PI * Math.pow(s, plan.bluntness)), plan.fullness);
    return Math.max(body, 0.1 * s);
  };

  b.grid(16, 28, (around, along) => {
    const x = NOSE - along * bodyLength;
    const profile = outline(x);
    const theta = around * Math.PI * 2;
    const h = plan.halfHeight * profile;
    const w = plan.halfWidth * Math.pow(profile, 0.8);
    const [c, s] = [Math.cos(theta), Math.sin(theta)];
    const nearNose = along < 0.12 ? (0.12 - along) * 5 : 0;
    return {
      p: [x, s * h, c * w],
      n: normalize([
        nearNose,
        s / Math.max(plan.halfHeight, 0.01),
        c / Math.max(plan.halfWidth, 0.01),
      ]),
      u: NOSE - x,
      v: s,
      extra: 0,
    };
  });

  const median = (root: (s: number) => Vec3, tip: (s: number) => Vec3, columns: number): void => {
    b.grid(columns, 4, (s, t) => {
      const [r, e] = [root(s), tip(s)];
      const p: Vec3 = [mix(r[0], e[0], t), mix(r[1], e[1], t), mix(r[2], e[2], t)];
      return { p, n: [0, 0, 1], u: NOSE - p[0], v: t, extra: Math.max(t, 0.02) };
    });
  };

  const { tail, dorsal, anal } = plan;
  const tailStart = plan.bodyEnd + 0.03;
  median(
    (s) => [tailStart, (s * 2 - 1) * tail.rootHeight, 0],
    (s) => {
      const q = s * 2 - 1;
      return [-0.5 + tail.fork * (1 - Math.abs(q)), q * tail.spread, 0];
    },
    12,
  );

  const edgeFin = (fin: Fin, sign: number): void => {
    median(
      (s) => {
        const x = mix(fin.from, fin.to, s);
        return [x, sign * plan.halfHeight * outline(x) * 0.93, 0];
      },
      (s) => {
        const x = mix(fin.from, fin.to, s);
        return [
          x - fin.sweep * (0.25 + s),
          sign *
            (plan.halfHeight * outline(x) * 0.93 + fin.height * finShape(s, fin.pointed ?? false)),
          0,
        ];
      },
      10,
    );
  };
  edgeFin(dorsal, 1);
  edgeFin(anal, -1);

  for (const side of [1, -1]) {
    const size = plan.pectoral;
    const y = -plan.halfHeight * 0.32;
    const z = side * plan.halfWidth * 0.82;
    b.grid(4, 3, (s, t) => {
      const x = 0.26 - s * size * 0.45;
      const p: Vec3 = [
        x - t * size * 0.9,
        y - t * size * 0.55,
        z + side * t * size * (0.5 + s * 0.2),
      ];
      return { p, n: normalize([0, 0.4, side]), u: NOSE - p[0], v: t, extra: Math.max(t, 0.02) };
    });
  }
  return b;
}

/** Per species: wave count, amplitude and species index, then the eye's position and radius. */
const PARAMS_GLSL = `
layout(std140) uniform Params {
  vec4 uShape;
  vec4 uEye;
};
`;

const VS = `${FRAME_GLSL}${PARAMS_GLSL}
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aUv;
layout(location = 3) in float aFin;
layout(location = 4) in vec4 iPosScale;
layout(location = 5) in vec4 iForwardPhase;
layout(location = 6) in vec4 iUpTurn;
layout(location = 7) in float iVariation;
out vec3 vP; out vec3 vN; out vec3 vL; out vec2 vU; out float vFin; out float vVariation;
void main() {
  float uWave = uShape.x;
  float uAmplitude = uShape.y;
  vec3 p = aPos;
  float u = clamp(aUv.x, 0.0, 1.1);
  float envelope = u * u + 0.05;
  float phase = iForwardPhase.w;
  float offset = sin(u * uWave - phase);
  float slope = cos(u * uWave - phase) * uWave * envelope;
  p.z += uAmplitude * envelope * offset;
  float behind = max(0.2 - aPos.x, 0.0);
  p.z += iUpTurn.w * behind * behind * 0.9;

  vec3 F = iForwardPhase.xyz;
  vec3 U = iUpTurn.xyz;
  vec3 S = cross(F, U);
  vec3 n = aNormal;
  n.x += slope * uAmplitude * n.z;

  vec3 world = iPosScale.xyz + iPosScale.w * (p.x * F + p.y * U + p.z * S);
  vP = world;
  vN = normalize(n.x * F + n.y * U + n.z * S);
  vL = aPos;
  vU = aUv;
  vFin = aFin;
  vVariation = iVariation;
  gl_Position = uViewProj * vec4(world, 1.0);
}`;

const FS = `${FRAME_GLSL}${PARAMS_GLSL}${NOISE_GLSL}${WATER_GLSL}
in vec3 vP; in vec3 vN; in vec3 vL; in vec2 vU; in float vFin; in float vVariation;
out vec4 outColor;

float band(float x, float center, float halfWidth, float soft) {
  return 1.0 - smoothstep(halfWidth, halfWidth + soft, abs(x - center));
}

void main() {
  int uSpecies = int(uShape.z + 0.5);
  vec3 N = normalize(vN);
  vec3 V = normalize(uCamPos - vP);
  if (dot(N, V) < 0.0) N = -N;
  float NoV = max(dot(N, V), 0.0);
  float u = vU.x;
  float v = vU.y;
  bool fin = vFin > 0.001;

  vec3 albedo = vec3(0.6);
  vec3 glow = vec3(0.0);
  float alpha = 1.0;
  float gloss = 60.0;
  float spec = 0.55;

  float scales = 0.92 + 0.08 * sin(u * 150.0 + sin(v * 22.0) * 2.0) * sin(v * 70.0 + u * 40.0);

  if (fin) {
    float rays = 0.55 + 0.45 * sin(vL.y * 130.0 + vL.x * 60.0);
    alpha = mix(0.82, 0.22, vFin) * (0.78 + 0.22 * rays);
    spec = 0.2;
    if (uSpecies == 0) {
      albedo = vec3(0.78, 0.85, 0.88);
    } else if (uSpecies == 1) {
      albedo = vec3(0.95, 0.32, 0.1);
      alpha = mix(alpha, 0.85, 0.3);
    } else if (uSpecies == 2) {
      albedo = mix(vec3(0.75, 0.72, 0.6), vec3(0.08), smoothstep(0.82, 0.98, vFin) * 0.8);
      float streak = band(fract(vL.x * 9.0 + vFin * 0.9), 0.5, 0.08, 0.1);
      albedo = mix(albedo, vec3(0.14), streak * smoothstep(0.1, 0.5, vFin) * 0.5);
      alpha = mix(0.6, 0.16, vFin);
    } else if (uSpecies == 3) {
      albedo = mix(vec3(0.95, 0.42, 0.14), vec3(0.14, 0.75, 0.82), vFin);
      alpha = 0.78;
    } else {
      albedo = vec3(0.7, 0.65, 0.5);
      float spots = step(0.72, hash13(floor(vL * 80.0)));
      albedo *= 1.0 - spots * 0.65;
    }
  } else if (uSpecies == 0) {
    vec3 back = vec3(0.3, 0.3, 0.2);
    vec3 belly = vec3(0.83, 0.86, 0.87);
    albedo = mix(belly, back, smoothstep(0.1, 0.8, v));
    float neonMask = band(v, 0.27, 0.13, 0.08) * band(u, 0.36, 0.3, 0.05);
    vec3 neon = mix(vec3(0.05, 0.48, 1.0), vec3(0.1, 1.0, 0.86), 0.5 + 0.5 * sin(u * 16.0 + (1.0 - NoV) * 7.0));
    albedo = mix(albedo, neon * 0.65, neonMask);
    glow += neon * neonMask * 0.85;
    float redMask = smoothstep(0.1, -0.1, v) * band(u, 0.59, 0.17, 0.04);
    albedo = mix(albedo, vec3(0.85, 0.05, 0.05), redMask);
    glow += vec3(0.5, 0.02, 0.03) * redMask * 0.6;
    albedo *= scales;
  } else if (uSpecies == 1) {
    albedo = mix(vec3(1.0, 0.36, 0.08), vec3(0.62, 0.15, 0.06), smoothstep(-0.2, 0.9, v));
    albedo = mix(albedo, vec3(1.0, 0.6, 0.25), band(v, 0.08, 0.08, 0.12) * 0.6);
    glow += vec3(0.4, 0.08, 0.01) * 0.65;
    albedo *= scales;
  } else if (uSpecies == 2) {
    albedo = mix(vec3(0.92, 0.9, 0.78), vec3(0.8, 0.78, 0.66), smoothstep(-0.8, 0.8, v));
    albedo = mix(albedo, vec3(0.86, 0.64, 0.28), smoothstep(0.26, 0.05, u) * 0.75);
    float bar1 = band(u + v * 0.02, 0.27, 0.018, 0.012);
    float bar2 = band(u + v * 0.04, 0.46, 0.028, 0.012);
    float bar3 = band(u - v * 0.02, 0.66, 0.018, 0.012);
    albedo = mix(albedo, vec3(0.03), max(max(bar1, bar2), bar3) * 0.92);
    albedo *= scales;
  } else if (uSpecies == 3) {
    float flow = sin(v * 9.0 + sin(u * 7.0 + vVariation * 8.0) * 1.4 + vVariation * 5.0);
    vec3 warm = mix(vec3(0.95, 0.37, 0.09), vec3(0.8, 0.16, 0.1), vVariation);
    vec3 cool = vec3(0.1, 0.72, 0.8);
    albedo = mix(warm, cool, smoothstep(0.35, 0.8, flow));
    float rim = smoothstep(0.86, 1.0, abs(v));
    albedo = mix(albedo, vec3(0.1, 0.8, 0.9), rim * 0.8);
    albedo = mix(albedo, vec3(0.08, 0.05, 0.03), band(u, 0.2, 0.02, 0.018) * 0.85);
    albedo *= 0.96 + 0.08 * sin(u * 180.0);
    glow += cool * smoothstep(0.35, 0.8, flow) * 0.06;
  } else {
    albedo = mix(vec3(0.9, 0.84, 0.72), vec3(0.42, 0.37, 0.28), smoothstep(-0.5, 0.5, v));
    float cell = hash13(floor(vec3(vL.xy * 64.0, vVariation * 20.0)));
    albedo *= 1.0 - step(0.72, cell) * step(-0.4, v) * 0.72;
    albedo *= 0.94 + 0.06 * sin(u * 110.0);
  }

  if (!fin) {
    vec2 e = (vL.xy - uEye.xy) / uEye.z;
    float r = length(e);
    if (r < 1.0) {
      albedo = mix(vec3(0.82, 0.72, 0.42), vec3(0.008), smoothstep(0.75, 0.55, r));
      glow = vec3(0.0);
      spec = 1.2;
      gloss = 200.0;
    }
    // Silver sides flash as fish turn, which sells the "school in motion" look.
    glow += (uSpecies < 2 ? 0.1 : 0.05) * uLightCol * pow(1.0 - NoV, 3.0) * vec3(0.7, 0.9, 1.0);
  }

  vec3 col = shade(albedo, N, vP, gloss, spec, fin ? 0.6 : 0.08);
  col += glow * mix(1.3, 0.75, uDay);
  outColor = vec4(applyWater(col, vP), alpha);
}`;

const WGSL = `${FRAME_WGSL}${NOISE_WGSL}${WATER_WGSL}
struct Params { shape: vec4f, eye: vec4f }
@group(1) @binding(0) var<uniform> params: Params;

struct VIn {
  @location(0) pos: vec3f,
  @location(1) normal: vec3f,
  @location(2) uv: vec2f,
  @location(3) fin: f32,
  @location(4) posScale: vec4f,
  @location(5) forwardPhase: vec4f,
  @location(6) upTurn: vec4f,
  @location(7) variation: f32,
}
struct VOut {
  @builtin(position) clip: vec4f,
  @location(0) p: vec3f,
  @location(1) n: vec3f,
  @location(2) l: vec3f,
  @location(3) u: vec2f,
  @location(4) fin: f32,
  @location(5) variation: f32,
}

@vertex fn vs(a: VIn) -> VOut {
  let wave = params.shape.x;
  let amplitude = params.shape.y;
  var p = a.pos;
  let u = clamp(a.uv.x, 0.0, 1.1);
  let envelope = u * u + 0.05;
  let phase = a.forwardPhase.w;
  let offset = sin(u * wave - phase);
  let slope = cos(u * wave - phase) * wave * envelope;
  p.z += amplitude * envelope * offset;
  let behind = max(0.2 - a.pos.x, 0.0);
  p.z += a.upTurn.w * behind * behind * 0.9;

  let F = a.forwardPhase.xyz;
  let U = a.upTurn.xyz;
  let S = cross(F, U);
  var n = a.normal;
  n.x += slope * amplitude * n.z;

  let world = a.posScale.xyz + a.posScale.w * (p.x * F + p.y * U + p.z * S);
  var o: VOut;
  o.p = world;
  o.n = normalize(n.x * F + n.y * U + n.z * S);
  o.l = a.pos;
  o.u = a.uv;
  o.fin = a.fin;
  o.variation = a.variation;
  o.clip = project(world);
  return o;
}

fn band(x: f32, center: f32, halfWidth: f32, soft: f32) -> f32 {
  return 1.0 - smoothstep(halfWidth, halfWidth + soft, abs(x - center));
}

@fragment fn fs(f: VOut) -> @location(0) vec4f {
  let species = i32(params.shape.z + 0.5);
  var N = normalize(f.n);
  let V = normalize(frame.camPos - f.p);
  if (dot(N, V) < 0.0) { N = -N; }
  let NoV = max(dot(N, V), 0.0);
  let u = f.u.x;
  let v = f.u.y;
  let fin = f.fin > 0.001;

  var albedo = vec3f(0.6);
  var glow = vec3f(0.0);
  var alpha = 1.0;
  var gloss = 60.0;
  var spec = 0.55;

  let scales = 0.92 + 0.08 * sin(u * 150.0 + sin(v * 22.0) * 2.0) * sin(v * 70.0 + u * 40.0);

  if (fin) {
    let rays = 0.55 + 0.45 * sin(f.l.y * 130.0 + f.l.x * 60.0);
    alpha = mix(0.82, 0.22, f.fin) * (0.78 + 0.22 * rays);
    spec = 0.2;
    if (species == 0) {
      albedo = vec3f(0.78, 0.85, 0.88);
    } else if (species == 1) {
      albedo = vec3f(0.95, 0.32, 0.1);
      alpha = mix(alpha, 0.85, 0.3);
    } else if (species == 2) {
      albedo = mix(vec3f(0.75, 0.72, 0.6), vec3f(0.08), smoothstep(0.82, 0.98, f.fin) * 0.8);
      let streak = band(fract(f.l.x * 9.0 + f.fin * 0.9), 0.5, 0.08, 0.1);
      albedo = mix(albedo, vec3f(0.14), streak * smoothstep(0.1, 0.5, f.fin) * 0.5);
      alpha = mix(0.6, 0.16, f.fin);
    } else if (species == 3) {
      albedo = mix(vec3f(0.95, 0.42, 0.14), vec3f(0.14, 0.75, 0.82), f.fin);
      alpha = 0.78;
    } else {
      albedo = vec3f(0.7, 0.65, 0.5);
      let spots = step(0.72, hash13(floor(f.l * 80.0)));
      albedo *= 1.0 - spots * 0.65;
    }
  } else if (species == 0) {
    let back = vec3f(0.3, 0.3, 0.2);
    let belly = vec3f(0.83, 0.86, 0.87);
    albedo = mix(belly, back, smoothstep(0.1, 0.8, v));
    let neonMask = band(v, 0.27, 0.13, 0.08) * band(u, 0.36, 0.3, 0.05);
    let neon = mix(vec3f(0.05, 0.48, 1.0), vec3f(0.1, 1.0, 0.86), 0.5 + 0.5 * sin(u * 16.0 + (1.0 - NoV) * 7.0));
    albedo = mix(albedo, neon * 0.65, neonMask);
    glow += neon * neonMask * 0.85;
    let redMask = smoothstep(0.1, -0.1, v) * band(u, 0.59, 0.17, 0.04);
    albedo = mix(albedo, vec3f(0.85, 0.05, 0.05), redMask);
    glow += vec3f(0.5, 0.02, 0.03) * redMask * 0.6;
    albedo *= scales;
  } else if (species == 1) {
    albedo = mix(vec3f(1.0, 0.36, 0.08), vec3f(0.62, 0.15, 0.06), smoothstep(-0.2, 0.9, v));
    albedo = mix(albedo, vec3f(1.0, 0.6, 0.25), band(v, 0.08, 0.08, 0.12) * 0.6);
    glow += vec3f(0.4, 0.08, 0.01) * 0.65;
    albedo *= scales;
  } else if (species == 2) {
    albedo = mix(vec3f(0.92, 0.9, 0.78), vec3f(0.8, 0.78, 0.66), smoothstep(-0.8, 0.8, v));
    albedo = mix(albedo, vec3f(0.86, 0.64, 0.28), smoothstep(0.26, 0.05, u) * 0.75);
    let bar1 = band(u + v * 0.02, 0.27, 0.018, 0.012);
    let bar2 = band(u + v * 0.04, 0.46, 0.028, 0.012);
    let bar3 = band(u - v * 0.02, 0.66, 0.018, 0.012);
    albedo = mix(albedo, vec3f(0.03), max(max(bar1, bar2), bar3) * 0.92);
    albedo *= scales;
  } else if (species == 3) {
    let flow = sin(v * 9.0 + sin(u * 7.0 + f.variation * 8.0) * 1.4 + f.variation * 5.0);
    let warm = mix(vec3f(0.95, 0.37, 0.09), vec3f(0.8, 0.16, 0.1), f.variation);
    let cool = vec3f(0.1, 0.72, 0.8);
    albedo = mix(warm, cool, smoothstep(0.35, 0.8, flow));
    let rim = smoothstep(0.86, 1.0, abs(v));
    albedo = mix(albedo, vec3f(0.1, 0.8, 0.9), rim * 0.8);
    albedo = mix(albedo, vec3f(0.08, 0.05, 0.03), band(u, 0.2, 0.02, 0.018) * 0.85);
    albedo *= 0.96 + 0.08 * sin(u * 180.0);
    glow += cool * smoothstep(0.35, 0.8, flow) * 0.06;
  } else {
    albedo = mix(vec3f(0.9, 0.84, 0.72), vec3f(0.42, 0.37, 0.28), smoothstep(-0.5, 0.5, v));
    let cell = hash13(floor(vec3f(f.l.xy * 64.0, f.variation * 20.0)));
    albedo *= 1.0 - step(0.72, cell) * step(-0.4, v) * 0.72;
    albedo *= 0.94 + 0.06 * sin(u * 110.0);
  }

  if (!fin) {
    let e = (f.l.xy - params.eye.xy) / params.eye.z;
    let r = length(e);
    if (r < 1.0) {
      albedo = mix(vec3f(0.82, 0.72, 0.42), vec3f(0.008), smoothstep(0.75, 0.55, r));
      glow = vec3f(0.0);
      spec = 1.2;
      gloss = 200.0;
    }
    // Silver sides flash as fish turn, which sells the "school in motion" look.
    glow += select(0.05, 0.1, species < 2) * frame.lightCol * pow(1.0 - NoV, 3.0) * vec3f(0.7, 0.9, 1.0);
  }

  var col = shade(albedo, N, f.p, gloss, spec, select(0.08, 0.6, fin));
  col += glow * mix(1.3, 0.75, frame.day);
  return vec4f(applyWater(col, f.p), alpha);
}`;

const FLOATS_PER_FISH = 13;
const INSTANCE_LAYOUT = [4, 4, 4, 1];
const PARAMS_BYTES = 32;

export class FishRenderer implements Drawable {
  private readonly pipeline: Pipeline;
  private readonly shadow: Pipeline;
  private readonly meshes: Mesh[];
  private readonly buffers: Floats[];
  private readonly counts: number[];
  private readonly params: Floats = new Float32Array(PARAMS_BYTES / 4);

  constructor(device: Device, targets: SceneTargets) {
    const source = shader(VS, FS, WGSL);
    const extra = { instances: INSTANCE_LAYOUT, params: PARAMS_BYTES };
    // Alpha to coverage lets fins fade out without sorting them.
    this.pipeline = scenePipeline(device, targets, "fish", source, {
      ...extra,
      alphaToCoverage: true,
    });
    this.shadow = shadowPipeline(device, "fish", source, extra);
    this.meshes = PLANS.map((plan) => device.createMesh(buildFish(plan).build(), INSTANCE_LAYOUT));
    this.buffers = SPECIES.map((def) => new Float32Array(def.count * FLOATS_PER_FISH));
    this.counts = SPECIES.map(() => 0);
  }

  update(fish: readonly Fish[]): void {
    this.counts.fill(0);
    for (const f of fish) {
      const buffer = this.buffers[f.species];
      const i = this.counts[f.species]++ * FLOATS_PER_FISH;
      buffer[i] = f.x;
      buffer[i + 1] = f.y;
      buffer[i + 2] = f.z;
      buffer[i + 3] = f.scale;
      buffer[i + 4] = f.fx;
      buffer[i + 5] = f.fy;
      buffer[i + 6] = f.fz;
      buffer[i + 7] = f.phase;
      buffer[i + 8] = f.ux;
      buffer[i + 9] = f.uy;
      buffer[i + 10] = f.uz;
      buffer[i + 11] = f.turn;
      buffer[i + 12] = f.variation;
    }
    this.meshes.forEach((mesh, species) =>
      mesh.setInstances(this.buffers[species], this.counts[species]),
    );
  }

  draw({ device, pass, frame, water, shadow }: DrawContext): void {
    const pipeline = shadow ? this.shadow : this.pipeline;
    const { params } = this;
    this.meshes.forEach((mesh, species) => {
      const plan = PLANS[species];
      params.set([plan.wave, plan.amplitude, species, 0, plan.eye[0], plan.eye[1], plan.eye[2], 0]);
      pass.draw(pipeline, mesh, { frame, params: device.uniforms(params), textures: water });
    });
  }
}
