/** Everything outside the water: the dim room, the cabinet, the rims, the hood and its lamp. */

import { MeshBuilder } from "../lib/geometry";
import { Mesh, Program } from "../lib/gl";
import {
  bindWaterSamplers,
  FRAME_GLSL,
  NOISE_GLSL,
  TANK,
  WATER_GLSL,
  type Drawable,
} from "./common";

const WALL = 0;
const FLOOR = 1;
const CABINET = 2;
const BLACK = 3;
const LAMP = 4;
const BACKDROP = 5;

const FLOOR_Y = -1.52;

const VS = `${FRAME_GLSL}
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 3) in float aMaterial;
out vec3 vP; out vec3 vN; flat out int vMaterial;
void main() {
  vP = aPos; vN = aNormal; vMaterial = int(aMaterial + 0.5);
  gl_Position = uViewProj * vec4(aPos, 1.0);
}`;

const FS = `${FRAME_GLSL}${NOISE_GLSL}${WATER_GLSL}
in vec3 vP; in vec3 vN; flat in int vMaterial;
out vec4 outColor;

const vec2 CABINET_HALF = vec2(1.64, 0.8);

void main() {
  vec3 P = vP;
  vec3 N = normalize(vN);
  vec3 V = normalize(uCamPos - P);

  if (vMaterial == ${LAMP}) {
    outColor = vec4(uLightCol * 1.5, 1.0);
    return;
  }
  if (vMaterial == ${BACKDROP}) {
    float v = smoothstep(0.0, 1.0, P.y / WATER_Y);
    float clouds = vnoise(vec3(P.x * 1.8, P.y * 2.6, uTime * 0.02));
    vec3 col = mix(vec3(0.006, 0.018, 0.026), vec3(0.03, 0.12, 0.15), v * (0.6 + 0.6 * clouds));
    col *= 0.25 + length(uLightCol) * 0.33;
    outColor = vec4(applyWater(col, P), 1.0);
    return;
  }

  // The tank is the room's main light, so light falls off from it.
  vec3 toTank = vec3(0.0, 0.8, 0.0) - P;
  float d2 = dot(toTank, toTank);
  vec3 toTankN = toTank / sqrt(d2);
  vec3 tankCol = uLightCol * 0.3 + uScatterCol * 1.7;
  float glow = (0.16 + 0.84 * max(dot(N, toTankN), 0.0)) / (1.0 + d2 * 0.52);
  vec3 ambient = mix(vec3(0.007, 0.009, 0.014), vec3(0.042, 0.04, 0.042), uDay);

  vec3 albedo;
  float gloss = 0.0;
  if (vMaterial == ${WALL}) {
    albedo = vec3(0.26, 0.28, 0.29) * (0.9 + 0.16 * vnoise(P * 3.0) + 0.05 * vnoise(P * 40.0));
    // Light spilling out of the hood onto the wall behind the tank.
    vec2 fromHood = P.xy - vec2(0.0, 1.5);
    float halo = exp(-dot(fromHood, fromHood) * 0.32);
    ambient += uLightCol * 0.045 * halo;
  } else if (vMaterial == ${FLOOR}) {
    float plank = floor(P.x * 2.6);
    float grain = vnoise(vec3(P.x * 60.0, plank * 7.0, P.z * 3.5));
    float seam = smoothstep(0.0, 0.015, abs(fract(P.x * 2.6) - 0.5) - 0.48);
    albedo = mix(vec3(0.075, 0.05, 0.034), vec3(0.17, 0.115, 0.074), grain * 0.6 + hash13(vec3(plank)) * 0.4);
    albedo *= 1.0 - seam * 0.6;
    gloss = 0.32;
    // Contact shadow around the cabinet.
    vec2 q = abs(P.xz) - CABINET_HALF;
    float dist = length(max(q, 0.0)) + min(max(q.x, q.y), 0.0);
    glow *= 0.22 + 0.78 * smoothstep(0.0, 0.8, dist);
    ambient *= 0.3 + 0.7 * smoothstep(0.0, 0.55, dist);
  } else if (vMaterial == ${CABINET}) {
    float rings = vnoise(vec3(P.x * 2.0, P.y * 38.0 + vnoise(P * 2.0) * 7.0, P.z * 2.0));
    albedo = mix(vec3(0.03, 0.017, 0.01), vec3(0.14, 0.077, 0.04), rings);
    gloss = 0.14;
    float reveal = smoothstep(0.0, 0.006, abs(P.x) - 0.008) * smoothstep(0.0, 0.006, abs(P.y + 0.78) - 0.006);
    albedo *= 0.3 + 0.7 * reveal;
  } else {
    albedo = vec3(0.012);
    gloss = 0.25;
  }

  vec3 col = albedo * (ambient + tankCol * glow * 1.45);
  if (gloss > 0.0) {
    vec3 R = reflect(-V, N);
    float highlight = pow(max(dot(R, toTankN), 0.0), 22.0);
    float fresnel = 0.04 + 0.96 * pow(1.0 - max(dot(N, V), 0.0), 5.0);
    col += tankCol * highlight * gloss * (0.6 + fresnel * 3.0) / (1.0 + d2 * 0.14);
    // The window-shaped sheen that keeps black plastic readable.
    float sheen = smoothstep(0.35, 1.0, R.y) * smoothstep(0.4, 0.95, abs(R.z));
    col += vec3(0.08, 0.09, 0.1) * sheen * fresnel * (0.3 + 0.7 * uDay);
  }
  outColor = vec4(applyWater(col, P), 1.0);
}`;

export class Room implements Drawable {
  private readonly program: Program;
  private readonly mesh: Mesh;

  constructor(gl: WebGL2RenderingContext) {
    this.program = new Program(gl, VS, FS);
    bindWaterSamplers(this.program);

    const b = new MeshBuilder();
    const { halfX: hx, halfZ: hz, height: h } = TANK;

    // Room shell: floor plus three walls, opening towards the camera.
    const [wx, wz, wy] = [9, 14, 6];
    const back = -3.2;
    b.quad(
      [-wx, FLOOR_Y, wz],
      [wx, FLOOR_Y, wz],
      [wx, FLOOR_Y, back],
      [-wx, FLOOR_Y, back],
      [0, 1, 0],
      FLOOR,
    );
    b.quad(
      [-wx, FLOOR_Y, back],
      [wx, FLOOR_Y, back],
      [wx, wy, back],
      [-wx, wy, back],
      [0, 0, 1],
      WALL,
    );
    b.quad(
      [-wx, FLOOR_Y, wz],
      [-wx, FLOOR_Y, back],
      [-wx, wy, back],
      [-wx, wy, wz],
      [1, 0, 0],
      WALL,
    );
    b.quad([wx, FLOOR_Y, back], [wx, FLOOR_Y, wz], [wx, wy, wz], [wx, wy, back], [-1, 0, 0], WALL);

    // Cabinet, with the tank sitting on top.
    b.box([-1.64, FLOOR_Y + 0.05, -0.8], [1.64, -0.045, 0.8], CABINET);
    b.box([-1.58, FLOOR_Y, -0.74], [1.58, FLOOR_Y + 0.05, 0.74], BLACK);

    // Rims.
    const t = 0.04;
    const rim = (y0: number, y1: number): void => {
      b.box([-hx - t, y0, hz], [hx + t, y1, hz + t], BLACK);
      b.box([-hx - t, y0, -hz - t], [hx + t, y1, -hz], BLACK);
      b.box([-hx - t, y0, -hz], [-hx, y1, hz], BLACK);
      b.box([hx, y0, -hz], [hx + t, y1, hz], BLACK);
    };
    rim(-0.045, 0.045);
    rim(h - 0.03, h + 0.015);

    // Hood and its lamp, sitting in the air gap above the water.
    b.box([-hx - 0.055, h + 0.015, -hz - 0.055], [hx + 0.055, h + 0.11, hz + 0.055], BLACK);
    b.box([-hx + 0.08, h - 0.012, -0.16], [hx - 0.08, h + 0.015, 0.16], LAMP);

    // The painted backdrop on the rear glass.
    const z = -hz + 0.002;
    b.quad([-hx, 0, z], [hx, 0, z], [hx, h, z], [-hx, h, z], [0, 0, 1], BACKDROP);

    this.mesh = new Mesh(gl, b.build());
  }

  draw(shadowPass: boolean): void {
    if (shadowPass) return;
    this.program.use();
    this.mesh.draw();
  }
}
