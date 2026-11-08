/** Renders a tileable, animated caustic pattern into a texture every frame. */

import type { Device, Pipeline, Texture } from "../lib/gpu/types";
import type { Floats } from "../lib/math";
import { FULLSCREEN_GLSL, FULLSCREEN_WGSL, PRELUDE_WGSL, shader } from "./common";

const SIZE = 512;

// The warps use whole-number frequencies, so the pattern tiles across the unit square.
const FS = `
in vec2 vUv;
layout(std140) uniform Params { vec4 uTime4; };
out vec4 outColor;
const float TAU = 6.2831853;

float web(vec2 uv, float t, float freq, float seed) {
  vec2 p = uv * TAU * freq;
  vec2 w = p;
  w += 0.85 * vec2(
    sin(p.y + t * 0.61 + seed) + 0.5 * sin(p.y * 2.0 - t * 0.37 + p.x),
    cos(p.x - t * 0.53 + seed) + 0.5 * cos(p.x * 3.0 + t * 0.29 - p.y));
  w += 0.48 * vec2(sin(w.y * 2.0 + t * 0.83), cos(w.x * 2.0 - t * 0.71));
  w += 0.2 * vec2(sin(w.y * 3.0 - t * 1.13), cos(w.x * 3.0 + t * 1.03));
  return abs(sin(w.x * 1.5 + w.y * 0.5) + sin(w.y * 1.5 - w.x * 0.5));
}

void main() {
  float t = uTime4.x;
  float a = web(vUv, t, 2.0, 0.0);
  float b = web(vUv + 0.37, t * 0.8, 3.0, 2.1);
  float c = exp(-a * 5.0) + 0.6 * exp(-b * 6.5);
  c += 1.5 * exp(-(a + b) * 3.2);
  outColor = vec4(clamp(c * 0.95, 0.0, 1.0));
}`;

const WGSL = `${PRELUDE_WGSL}${FULLSCREEN_WGSL}
@group(1) @binding(0) var<uniform> time4: vec4f;
const TAU = 6.2831853;

fn web(uv: vec2f, t: f32, freq: f32, seed: f32) -> f32 {
  let p = uv * TAU * freq;
  var w = p;
  w += 0.85 * vec2f(
    sin(p.y + t * 0.61 + seed) + 0.5 * sin(p.y * 2.0 - t * 0.37 + p.x),
    cos(p.x - t * 0.53 + seed) + 0.5 * cos(p.x * 3.0 + t * 0.29 - p.y));
  w += 0.48 * vec2f(sin(w.y * 2.0 + t * 0.83), cos(w.x * 2.0 - t * 0.71));
  w += 0.2 * vec2f(sin(w.y * 3.0 - t * 1.13), cos(w.x * 3.0 + t * 1.03));
  return abs(sin(w.x * 1.5 + w.y * 0.5) + sin(w.y * 1.5 - w.x * 0.5));
}

@fragment fn fs(v: FsOut) -> @location(0) vec4f {
  let t = time4.x;
  let a = web(v.uv, t, 2.0, 0.0);
  let b = web(v.uv + 0.37, t * 0.8, 3.0, 2.1);
  var c = exp(-a * 5.0) + 0.6 * exp(-b * 6.5);
  c += 1.5 * exp(-(a + b) * 3.2);
  return vec4f(clamp(c * 0.95, 0.0, 1.0));
}`;

export class Caustics {
  readonly texture: Texture;
  private readonly pipeline: Pipeline;
  private readonly params: Floats = new Float32Array(4);

  constructor(device: Device) {
    this.pipeline = device.createPipeline({
      label: "caustics",
      shader: shader(FULLSCREEN_GLSL, FS, WGSL),
      vertex: "none",
      params: 16,
      color: "rgba8",
    });
    this.texture = device.createTexture({
      label: "caustics",
      width: SIZE,
      height: SIZE,
      format: "rgba8",
      wrap: "repeat",
      mipmaps: true,
      // The sand sees the pattern at grazing angles, where plain mipmapping smears it.
      anisotropic: true,
    });
  }

  render(device: Device, time: number): void {
    this.params[0] = time;
    const pass = device.beginPass({
      label: "caustics",
      color: { target: this.texture, clear: [0, 0, 0, 0] },
    });
    pass.draw(this.pipeline, null, { params: device.uniforms(this.params) });
    pass.end();
    device.generateMipmaps(this.texture);
  }
}
