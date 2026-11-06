/** Billboards for suspended specks, the aerator's bubble stream, and food flakes. */

import { MeshBuilder } from "../lib/geometry";
import { Mesh, Program } from "../lib/gl";
import { mix, type Floats, type Vec3 } from "../lib/math";
import { createRng, type Rng } from "../lib/noise";
import type { Food } from "../sim/school";
import { bindWaterSamplers, FRAME_GLSL, TANK, WATER_GLSL, type Drawable } from "./common";
import { sandHeight } from "./layout";

const KIND_DUST = 0;
const KIND_BUBBLE = 1;
const KIND_FOOD = 2;

const VS = `${FRAME_GLSL}
layout(location = 0) in vec3 aPos;
layout(location = 4) in vec4 iPosSize;
layout(location = 5) in vec2 iKindSeed;
uniform vec3 uRight;
uniform vec3 uUp;
out vec2 vUv; out vec3 vP; flat out int vKind; out float vSeed;
void main() {
  vUv = aPos.xy;
  vKind = int(iKindSeed.x + 0.5);
  vSeed = iKindSeed.y;
  vec3 world = iPosSize.xyz + (uRight * aPos.x + uUp * aPos.y) * iPosSize.w;
  vP = world;
  gl_Position = uViewProj * vec4(world, 1.0);
}`;

const FS = `${FRAME_GLSL}${WATER_GLSL}
in vec2 vUv; in vec3 vP; flat in int vKind; in float vSeed;
out vec4 outColor;
void main() {
  float r2 = dot(vUv, vUv);
  if (r2 > 1.0) discard;
  float r = sqrt(r2);
  vec3 atten = exp(-max(WATER_Y - vP.y, 0.0) * ABSORB);
  vec3 col;
  float alpha;
  if (vKind == ${KIND_DUST}) {
    float a = pow(1.0 - r2, 2.0);
    float glint = 0.35 + 0.65 * sin(uTime * 0.8 + vSeed * 40.0) * sin(uTime * 0.8 + vSeed * 40.0);
    col = uLightCol * atten * (0.14 + causticAt(vP) * 0.5) + uScatterCol * 1.5;
    alpha = a * 0.38 * glint;
  } else if (vKind == ${KIND_BUBBLE}) {
    float rim = pow(r, 3.5);
    vec2 h = vUv - vec2(-0.34, 0.38);
    float spark = exp(-dot(h, h) * 42.0);
    float under = exp(-dot(vUv - vec2(0.25, -0.5), vUv - vec2(0.25, -0.5)) * 24.0) * 0.35;
    col = uLightCol * atten * (rim * 0.5 + spark * 1.8 + under) + uScatterCol * (1.2 * rim + 0.2);
    alpha = clamp(rim * 0.62 + spark + under + 0.05, 0.0, 1.0);
  } else {
    vec3 n = vec3(vUv, sqrt(1.0 - r2));
    float diffuse = 0.45 + 0.55 * max(dot(n, normalize(vec3(0.2, 0.9, 0.4))), 0.0);
    vec3 tone = mix(vec3(0.76, 0.28, 0.08), vec3(0.88, 0.62, 0.2), fract(vSeed * 13.7));
    col = tone * diffuse * (uLightCol * atten * 0.55 + uScatterCol * 2.0);
    alpha = 1.0 - smoothstep(0.82, 1.0, r);
  }
  col = applyWater(col, vP);
  outColor = vec4(col * alpha, alpha);
}`;

interface Speck {
  x: number;
  y: number;
  z: number;
  seed: number;
  size: number;
}

interface Bubble {
  x: number;
  y: number;
  z: number;
  size: number;
  speed: number;
  seed: number;
}

const DUST_COUNT = 300;
const MAX_BUBBLES = 90;
const MAX_FOOD = 90;
const FLOATS_PER = 6;
const AERATOR: Vec3 = [1.26, 0, -0.5];

export class Particles implements Drawable {
  private readonly gl: WebGL2RenderingContext;
  private readonly program: Program;
  private readonly mesh: Mesh;
  private readonly data: Floats = new Float32Array(
    (DUST_COUNT + MAX_BUBBLES + MAX_FOOD) * FLOATS_PER,
  );
  private readonly specks: Speck[] = [];
  private readonly bubbles: Bubble[] = [];
  private readonly rng: Rng = createRng(555);
  private emit = 0;
  private time = 0;
  private right: Vec3 = [1, 0, 0];
  private up: Vec3 = [0, 1, 0];

  constructor(gl: WebGL2RenderingContext) {
    this.gl = gl;
    this.program = new Program(gl, VS, FS);
    bindWaterSamplers(this.program);
    const builder = new MeshBuilder();
    builder.quad([-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0], [0, 0, 1]);
    this.mesh = new Mesh(gl, builder.build()).instanced([4, 2]);
    for (let i = 0; i < DUST_COUNT; i++) {
      this.specks.push({
        x: mix(-TANK.halfX, TANK.halfX, this.rng()),
        y: mix(0.1, TANK.water, this.rng()),
        z: mix(-TANK.halfZ, TANK.halfZ, this.rng()),
        seed: this.rng(),
        size: mix(0.0018, 0.0045, this.rng()),
      });
    }
  }

  update(dt: number, food: readonly Food[], right: Vec3, up: Vec3): void {
    this.right = right;
    this.up = up;
    this.time += dt;
    const { rng, data, time } = this;

    this.emit += dt * 17;
    while (this.emit >= 1 && this.bubbles.length < MAX_BUBBLES) {
      this.emit -= 1;
      const x = AERATOR[0] + (rng() - 0.5) * 0.035;
      const z = AERATOR[2] + (rng() - 0.5) * 0.035;
      this.bubbles.push({
        x,
        y: sandHeight(AERATOR[0], AERATOR[2]) + 0.02,
        z,
        size: mix(0.004, 0.013, rng() * rng()),
        speed: mix(0.32, 0.58, rng()),
        seed: rng() * 100,
      });
    }
    this.emit = Math.min(this.emit, 2);

    let count = 0;
    const write = (
      x: number,
      y: number,
      z: number,
      size: number,
      kind: number,
      seed: number,
    ): void => {
      const i = count++ * FLOATS_PER;
      data[i] = x;
      data[i + 1] = y;
      data[i + 2] = z;
      data[i + 3] = size;
      data[i + 4] = kind;
      data[i + 5] = seed;
    };

    for (const s of this.specks) {
      const phase = s.seed * 60;
      s.x += Math.sin(time * 0.09 + phase) * 0.008 * dt;
      s.y += (Math.sin(time * 0.13 + phase * 1.7) * 0.006 - 0.0015) * dt;
      s.z += Math.cos(time * 0.11 + phase * 0.6) * 0.008 * dt;
      if (s.y < 0.08) s.y = TANK.water - 0.02;
      write(s.x, s.y, s.z, s.size, KIND_DUST, s.seed);
    }

    for (let i = this.bubbles.length - 1; i >= 0; i--) {
      const b = this.bubbles[i];
      b.y += b.speed * dt;
      b.size += 0.0016 * dt;
      b.x += Math.sin(time * 6 + b.seed) * 0.05 * dt;
      b.z += Math.cos(time * 5 + b.seed * 1.3) * 0.04 * dt;
      if (b.y > TANK.water - b.size) {
        this.bubbles.splice(i, 1);
        continue;
      }
      write(b.x, b.y, b.z, b.size, KIND_BUBBLE, b.seed);
    }

    let flakes = 0;
    for (const f of food) {
      if (f.age < 0 || flakes++ >= MAX_FOOD) continue;
      write(f.x, f.y, f.z, 0.0065, KIND_FOOD, f.drift);
    }

    this.mesh.setInstances(data, count);
  }

  draw(shadowPass: boolean): void {
    if (shadowPass) return;
    const { gl, right, up } = this;
    this.program
      .use()
      .vec3("uRight", right[0], right[1], right[2])
      .vec3("uUp", up[0], up[1], up[2]);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    this.mesh.draw();
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  }
}
