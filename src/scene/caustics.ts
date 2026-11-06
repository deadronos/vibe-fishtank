/** Renders a tileable, animated caustic pattern into a texture every frame. */

import { createTarget, drawFullscreen, FULLSCREEN_VS, Program, type Target } from "../lib/gl";
import { UNIT_CAUSTICS } from "./common";

const SIZE = 512;

// The warps use whole-number frequencies, so the pattern tiles across the unit square.
const FS = `
in vec2 vUv;
uniform float uTime;
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
  float t = uTime;
  float a = web(vUv, t, 2.0, 0.0);
  float b = web(vUv + 0.37, t * 0.8, 3.0, 2.1);
  float c = exp(-a * 5.0) + 0.6 * exp(-b * 6.5);
  c += 1.5 * exp(-(a + b) * 3.2);
  outColor = vec4(clamp(c * 0.95, 0.0, 1.0));
}`;

export class Caustics {
  readonly target: Target;
  private readonly gl: WebGL2RenderingContext;
  private readonly program: Program;

  constructor(gl: WebGL2RenderingContext) {
    this.gl = gl;
    this.program = new Program(gl, FULLSCREEN_VS, FS);
    this.target = createTarget(
      gl,
      SIZE,
      SIZE,
      { internal: gl.RGBA8, type: gl.UNSIGNED_BYTE },
      { repeat: true, mipmaps: true },
    );
    gl.activeTexture(gl.TEXTURE0 + UNIT_CAUSTICS);
    gl.bindTexture(gl.TEXTURE_2D, this.target.texture);
  }

  render(time: number): void {
    const { gl } = this;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.target.framebuffer);
    gl.viewport(0, 0, SIZE, SIZE);
    gl.disable(gl.DEPTH_TEST);
    this.program.use().float("uTime", time);
    drawFullscreen(gl);
    gl.activeTexture(gl.TEXTURE0 + UNIT_CAUSTICS);
    gl.generateMipmap(gl.TEXTURE_2D);
  }
}
