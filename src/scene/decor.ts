/** Hardscape: craggy stones, pebbles and a branching piece of driftwood, baked into one mesh. */

import { icosphere, MeshBuilder } from "../lib/geometry";
import { Mesh, Program } from "../lib/gl";
import { mix, type Vec3 } from "../lib/math";
import { createRng, fbm3 } from "../lib/noise";
import {
  bindWaterSamplers,
  FRAME_GLSL,
  NOISE_GLSL,
  SHADOW_FS,
  TANK,
  WATER_GLSL,
  type Drawable,
} from "./common";
import { DRIFTWOOD, sandHeight, STONES, stoneClearance } from "./layout";

const STONE = 0;
const WOOD = 1;
const PEBBLE = 2;

const VS = `${FRAME_GLSL}
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aUv;
layout(location = 3) in float aKind;
out vec3 vP; out vec3 vN; out vec2 vUv; out float vKind;
void main() {
  vP = aPos; vN = aNormal; vUv = aUv; vKind = aKind;
  gl_Position = uViewProj * vec4(aPos, 1.0);
}`;

const FS = `${FRAME_GLSL}${NOISE_GLSL}${WATER_GLSL}
in vec3 vP; in vec3 vN; in vec2 vUv; in float vKind;
out vec4 outColor;
void main() {
  vec3 N = normalize(vN);
  vec3 P = vP;
  vec3 albedo;
  float spec = 0.08;
  if (vKind < 0.5) {
    float layers = vnoise(vec3(P.x * 5.0, P.y * 26.0 + P.x * 3.0, P.z * 5.0));
    float fine = vnoise(P * 44.0);
    albedo = mix(vec3(0.07, 0.07, 0.075), vec3(0.3, 0.27, 0.235), layers * 0.65 + fine * 0.35);
    float moss = smoothstep(0.55, 0.92, N.y + (vnoise(P * 15.0) - 0.5) * 0.9);
    albedo = mix(albedo, vec3(0.2, 0.36, 0.1) * (0.6 + fine * 0.8), moss * 0.85);
    N = normalize(N + (vec3(vnoise(P * 60.0), vnoise(P.zxy * 60.0), vnoise(P.yzx * 60.0)) - 0.5) * 0.28);
  } else if (vKind < 1.5) {
    float grain = vnoise(vec3(vUv.x * 18.0, vUv.y * 5.0, 3.0));
    float streak = vnoise(vec3(vUv.x * 60.0, vUv.y * 9.0, 8.0));
    albedo = mix(vec3(0.09, 0.055, 0.03), vec3(0.34, 0.22, 0.12), grain * 0.55 + streak * 0.45);
    N = normalize(N + (vec3(streak, grain, streak * grain) - 0.5) * 0.35);
  } else {
    float tone = hash13(floor(P * 11.0));
    albedo = mix(vec3(0.3, 0.27, 0.24), vec3(0.72, 0.64, 0.52), tone);
    spec = 0.2;
  }
  vec3 col = shade(albedo, N, P, 22.0, spec, 0.0);
  outColor = vec4(applyWater(col, P), 1.0);
}`;

const SHADOW_VS = `${FRAME_GLSL}
layout(location = 0) in vec3 aPos;
void main() { gl_Position = uViewProj * vec4(aPos, 1.0); }`;

function addStone(
  builder: MeshBuilder,
  sphere: ReturnType<typeof icosphere>,
  base: Vec3,
  radius: number,
  height: number,
  seed: number,
): void {
  const yaw = seed * 2.4;
  const [c, s] = [Math.cos(yaw), Math.sin(yaw)];
  const positions = sphere.positions.map((d): Vec3 => {
    const large = fbm3(d[0] * 1.6 + seed * 9, d[1] * 2.4, d[2] * 1.6 - seed * 5, 3);
    const ridge = 1 - Math.abs(fbm3(d[0] * 4 + seed, d[1] * 7, d[2] * 4, 4) * 2 - 1);
    const strata = Math.sin(d[1] * 26 + large * 8) * 0.018;
    const r = 0.58 + large * 0.68 + ridge * ridge * 0.2 + strata;
    const x = d[0] * r * radius;
    const z = d[2] * r * radius * 0.8;
    return [
      base[0] + x * c - z * s,
      base[1] + Math.max(d[1], -0.6) * r * height,
      base[2] + x * s + z * c,
    ];
  });
  builder.smooth(positions, sphere.indices, STONE);
}

function catmull(points: readonly Vec3[], subdivisions: number): Vec3[] {
  const out: Vec3[] = [];
  for (let i = 0; i < points.length - 1; i++) {
    const p0 = points[Math.max(0, i - 1)];
    const p1 = points[i];
    const p2 = points[i + 1];
    const p3 = points[Math.min(points.length - 1, i + 2)];
    for (let j = 0; j < subdivisions; j++) {
      const t = j / subdivisions;
      const t2 = t * t;
      const t3 = t2 * t;
      const at = (k: number): number =>
        0.5 *
        (2 * p1[k] +
          (p2[k] - p0[k]) * t +
          (2 * p0[k] - 5 * p1[k] + 4 * p2[k] - p3[k]) * t2 +
          (3 * p1[k] - p0[k] - 3 * p2[k] + p3[k]) * t3);
      out.push([at(0), at(1), at(2)]);
    }
  }
  out.push(points[points.length - 1]);
  return out;
}

export class Decor implements Drawable {
  private readonly program: Program;
  private readonly shadow: Program;
  private readonly mesh: Mesh;

  constructor(gl: WebGL2RenderingContext) {
    this.program = new Program(gl, VS, FS);
    this.shadow = new Program(gl, SHADOW_VS, SHADOW_FS);
    bindWaterSamplers(this.program);

    const builder = new MeshBuilder();
    const rock = icosphere(4);
    for (const stone of STONES) {
      const base: Vec3 = [stone.x, sandHeight(stone.x, stone.z) - 0.015, stone.z];
      addStone(builder, rock, base, stone.radius, stone.height, stone.seed);
    }

    DRIFTWOOD.forEach((branch, index) => {
      const points = catmull(branch, 7);
      const start = index === 0 ? 0.05 : 0.028;
      const radii = points.map((point, i) => {
        const t = i / (points.length - 1);
        const knots = 1 + 0.16 * fbm3(point[0] * 12, point[1] * 12, point[2] * 12, 2);
        return mix(start, 0.006, Math.pow(t, 0.8)) * knots;
      });
      builder.tube(points, radii, 10, WOOD);
    });

    const rng = createRng(1729);
    const pebble = icosphere(2);
    for (let i = 0; i < 150; i++) {
      const x = mix(-TANK.halfX + 0.05, TANK.halfX - 0.05, rng());
      const z = mix(-TANK.halfZ + 0.05, TANK.halfZ - 0.05, rng());
      if (stoneClearance(x, z) > 0.75 && rng() < 0.7) continue;
      const size = mix(0.006, 0.02, rng() * rng());
      const y = sandHeight(x, z);
      const stretch = 0.6 + rng() * 0.5;
      const positions = pebble.positions.map((d): Vec3 => [
        x + d[0] * size,
        y + d[1] * size * 0.65,
        z + d[2] * size * stretch,
      ]);
      builder.smooth(positions, pebble.indices, PEBBLE);
    }

    this.mesh = new Mesh(gl, builder.build());
  }

  draw(shadowPass: boolean): void {
    (shadowPass ? this.shadow : this.program).use();
    this.mesh.draw();
  }
}
