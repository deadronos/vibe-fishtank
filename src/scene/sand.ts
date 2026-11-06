/** The substrate: a height field, plus a skirt showing the sand's cross-section through the glass. */

import { MeshBuilder } from "../lib/geometry";
import { Mesh, Program } from "../lib/gl";
import { mix, normalize } from "../lib/math";
import {
  bindWaterSamplers,
  FRAME_GLSL,
  NOISE_GLSL,
  TANK,
  WATER_GLSL,
  type Drawable,
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

export class Sand implements Drawable {
  private readonly program: Program;
  private readonly mesh: Mesh;

  constructor(gl: WebGL2RenderingContext) {
    this.program = new Program(gl, VS, FS);
    bindWaterSamplers(this.program);

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
    this.mesh = new Mesh(gl, builder.build());
  }

  draw(shadowPass: boolean): void {
    if (shadowPass) return;
    this.program.use();
    this.mesh.draw();
  }
}
