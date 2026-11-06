/** The frame pipeline's screen-space half: MSAA resolve, light shafts, bloom and tone mapping. */

import {
  createTarget,
  deleteTarget,
  drawFullscreen,
  FULLSCREEN_VS,
  Program,
  type ColorFormat,
  type Target,
} from "../lib/gl";
import type { Floats } from "../lib/math";
import { bindWaterSamplers, FRAME_GLSL, UNIT_SHADOW, WATER_GLSL } from "./common";

const UNIT_A = 2;
const UNIT_B = 3;
const UNIT_C = 4;
const BLOOM_LEVELS = 5;

const SHAFTS_FS = `${FRAME_GLSL}${WATER_GLSL}
in vec2 vUv;
uniform sampler2D uDepth;
uniform mat4 uInvViewProj;
out vec4 outColor;

float ign(vec2 p) {
  return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
}

void main() {
  vec2 ndc = vUv * 2.0 - 1.0;
  vec4 farPoint = uInvViewProj * vec4(ndc, 1.0, 1.0);
  vec3 rd = normalize(farPoint.xyz / farPoint.w - uCamPos);
  vec2 hit = boxRay(uCamPos, rd);
  float a = max(hit.x, 0.0);
  float b = hit.y;
  if (b <= a) {
    outColor = vec4(0.0);
    return;
  }
  vec4 scene = uInvViewProj * vec4(ndc, texture(uDepth, vUv).r * 2.0 - 1.0, 1.0);
  b = min(b, length(scene.xyz / scene.w - uCamPos));
  if (b <= a) {
    outColor = vec4(0.0);
    return;
  }
  const int STEPS = 14;
  float stepLength = (b - a) / float(STEPS);
  float t = a + stepLength * ign(gl_FragCoord.xy);
  vec3 sum = vec3(0.0);
  for (int i = 0; i < STEPS; i++) {
    vec3 p = uCamPos + rd * t;
    float depth = max(WATER_Y - p.y, 0.0);
    // Blurred caustics, sampled along the light direction, make shafts that slowly shift.
    vec2 uv = lightUv(p);
    float beam = textureLod(uCaustics, uv * 0.42, 3.4).r + 0.5 * textureLod(uCaustics, uv * 0.19 + 0.3, 3.8).r;
    beam = smoothstep(0.08, 0.34, beam);
    vec3 lightPos = (uLightVP * vec4(p, 1.0)).xyz * 0.5 + 0.5;
    float visible = texture(uShadow, vec3(lightPos.xy, lightPos.z - 0.004));
    sum += beam * visible * exp(-depth * ABSORB) * exp(-(t - a) * FOG);
    t += stepLength;
  }
  outColor = vec4(sum * stepLength * uLightCol * 0.05, 1.0);
}`;

const DOWN_FS = `
in vec2 vUv;
uniform sampler2D uTex;
uniform vec2 uTexel;
uniform float uThreshold;
out vec4 outColor;
void main() {
  vec3 c = texture(uTex, vUv).rgb * 0.25;
  c += texture(uTex, vUv + uTexel * vec2(-1.0, -1.0)).rgb * 0.1875;
  c += texture(uTex, vUv + uTexel * vec2(1.0, -1.0)).rgb * 0.1875;
  c += texture(uTex, vUv + uTexel * vec2(-1.0, 1.0)).rgb * 0.1875;
  c += texture(uTex, vUv + uTexel * vec2(1.0, 1.0)).rgb * 0.1875;
  if (uThreshold > 0.0) {
    float l = max(c.r, max(c.g, c.b));
    c *= smoothstep(uThreshold * 0.6, uThreshold * 1.4, l);
    c = min(c, vec3(8.0));
  }
  outColor = vec4(c, 1.0);
}`;

const UP_FS = `
in vec2 vUv;
uniform sampler2D uTex;
uniform vec2 uTexel;
out vec4 outColor;
void main() {
  vec3 c = texture(uTex, vUv).rgb * 4.0;
  c += texture(uTex, vUv + uTexel * vec2(-1.0, 0.0)).rgb * 2.0;
  c += texture(uTex, vUv + uTexel * vec2(1.0, 0.0)).rgb * 2.0;
  c += texture(uTex, vUv + uTexel * vec2(0.0, -1.0)).rgb * 2.0;
  c += texture(uTex, vUv + uTexel * vec2(0.0, 1.0)).rgb * 2.0;
  c += texture(uTex, vUv + uTexel * vec2(-1.0, -1.0)).rgb;
  c += texture(uTex, vUv + uTexel * vec2(1.0, -1.0)).rgb;
  c += texture(uTex, vUv + uTexel * vec2(-1.0, 1.0)).rgb;
  c += texture(uTex, vUv + uTexel * vec2(1.0, 1.0)).rgb;
  outColor = vec4(c / 16.0, 1.0);
}`;

const COMPOSITE_FS = `
in vec2 vUv;
uniform sampler2D uScene;
uniform sampler2D uShafts;
uniform sampler2D uBloom;
uniform float uTime;
out vec4 outColor;

vec3 aces(vec3 x) {
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}
float noise(vec2 p) {
  return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}
void main() {
  vec3 hdr = texture(uScene, vUv).rgb + texture(uShafts, vUv).rgb + texture(uBloom, vUv).rgb * 0.2;
  vec2 q = vUv - 0.5;
  hdr *= 1.0 - dot(q, q) * 0.72;
  vec3 c = aces(hdr * 1.08);
  c = pow(c, vec3(1.0 / 2.2));
  c += (noise(gl_FragCoord.xy + fract(uTime) * 61.0) - 0.5) / 255.0 * 1.6;
  outColor = vec4(c, 1.0);
}`;

export class PostPipeline {
  private readonly gl: WebGL2RenderingContext;
  private readonly shafts: Program;
  private readonly down: Program;
  private readonly up: Program;
  private readonly composite: Program;
  private readonly format: ColorFormat;
  private readonly samples: number;

  private msaaFbo: WebGLFramebuffer | null = null;
  private colorRb: WebGLRenderbuffer | null = null;
  private depthRb: WebGLRenderbuffer | null = null;
  private resolveFbo: WebGLFramebuffer | null = null;
  private resolveColor: WebGLTexture | null = null;
  private resolveDepth: WebGLTexture | null = null;
  private shaftsTarget: Target | null = null;
  private bloomDown: Target[] = [];
  private bloomUp: Target[] = [];
  private width = 0;
  private height = 0;

  constructor(gl: WebGL2RenderingContext) {
    this.gl = gl;
    const float = gl.getExtension("EXT_color_buffer_float") !== null;
    this.format = float
      ? { internal: gl.RGBA16F, type: gl.HALF_FLOAT }
      : { internal: gl.RGBA8, type: gl.UNSIGNED_BYTE };
    const maxSamples = float
      ? gl.getInternalformatParameter(gl.RENDERBUFFER, gl.RGBA16F, gl.SAMPLES)
      : [4];
    const available = (maxSamples as Int32Array | number[])[0] ?? 0;
    this.samples = Math.min(4, available);

    this.shafts = new Program(gl, FULLSCREEN_VS, SHAFTS_FS);
    bindWaterSamplers(this.shafts);
    this.shafts.int("uDepth", UNIT_A);
    this.down = new Program(gl, FULLSCREEN_VS, DOWN_FS);
    this.down.use().int("uTex", UNIT_A);
    this.up = new Program(gl, FULLSCREEN_VS, UP_FS);
    this.up.use().int("uTex", UNIT_A);
    this.composite = new Program(gl, FULLSCREEN_VS, COMPOSITE_FS);
    this.composite.use().int("uScene", UNIT_A).int("uShafts", UNIT_B).int("uBloom", UNIT_C);
  }

  resize(width: number, height: number): void {
    if (width === this.width && height === this.height) return;
    this.width = width;
    this.height = height;
    this.release();
    const { gl, format, samples } = this;

    this.colorRb = gl.createRenderbuffer();
    gl.bindRenderbuffer(gl.RENDERBUFFER, this.colorRb);
    gl.renderbufferStorageMultisample(gl.RENDERBUFFER, samples, format.internal, width, height);
    this.depthRb = gl.createRenderbuffer();
    gl.bindRenderbuffer(gl.RENDERBUFFER, this.depthRb);
    gl.renderbufferStorageMultisample(
      gl.RENDERBUFFER,
      samples,
      gl.DEPTH_COMPONENT24,
      width,
      height,
    );
    this.msaaFbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.msaaFbo);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.RENDERBUFFER, this.colorRb);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, this.depthRb);

    gl.activeTexture(gl.TEXTURE7);
    this.resolveColor = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.resolveColor);
    gl.texStorage2D(gl.TEXTURE_2D, 1, format.internal, width, height);
    for (const [param, value] of [
      [gl.TEXTURE_MIN_FILTER, gl.LINEAR],
      [gl.TEXTURE_MAG_FILTER, gl.LINEAR],
      [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE],
      [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE],
    ] as const) {
      gl.texParameteri(gl.TEXTURE_2D, param, value);
    }
    this.resolveDepth = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.resolveDepth);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.DEPTH_COMPONENT24, width, height);
    for (const [param, value] of [
      [gl.TEXTURE_MIN_FILTER, gl.NEAREST],
      [gl.TEXTURE_MAG_FILTER, gl.NEAREST],
      [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE],
      [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE],
    ] as const) {
      gl.texParameteri(gl.TEXTURE_2D, param, value);
    }
    this.resolveFbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.resolveFbo);
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER,
      gl.COLOR_ATTACHMENT0,
      gl.TEXTURE_2D,
      this.resolveColor,
      0,
    );
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER,
      gl.DEPTH_ATTACHMENT,
      gl.TEXTURE_2D,
      this.resolveDepth,
      0,
    );

    this.shaftsTarget = createTarget(gl, Math.max(1, width >> 1), Math.max(1, height >> 1), format);
    let w = width;
    let h = height;
    for (let i = 0; i < BLOOM_LEVELS; i++) {
      w = Math.max(2, w >> 1);
      h = Math.max(2, h >> 1);
      this.bloomDown.push(createTarget(gl, w, h, format));
      if (i < BLOOM_LEVELS - 1) this.bloomUp.push(createTarget(gl, w, h, format));
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  private release(): void {
    const { gl } = this;
    gl.deleteFramebuffer(this.msaaFbo);
    gl.deleteRenderbuffer(this.colorRb);
    gl.deleteRenderbuffer(this.depthRb);
    gl.deleteFramebuffer(this.resolveFbo);
    gl.deleteTexture(this.resolveColor);
    gl.deleteTexture(this.resolveDepth);
    if (this.shaftsTarget) deleteTarget(gl, this.shaftsTarget);
    for (const target of [...this.bloomDown, ...this.bloomUp]) deleteTarget(gl, target);
    this.bloomDown = [];
    this.bloomUp = [];
  }

  /** Starts the HDR scene pass. */
  beginScene(): void {
    const { gl } = this;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.msaaFbo);
    gl.viewport(0, 0, this.width, this.height);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  }

  /** Resolves MSAA, computes shafts and bloom, and tone maps to the canvas. */
  finish(invViewProj: Floats, time: number): void {
    const { gl, width, height } = this;
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, this.msaaFbo);
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, this.resolveFbo);
    gl.blitFramebuffer(0, 0, width, height, 0, 0, width, height, gl.COLOR_BUFFER_BIT, gl.LINEAR);
    gl.blitFramebuffer(0, 0, width, height, 0, 0, width, height, gl.DEPTH_BUFFER_BIT, gl.NEAREST);

    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);

    const shafts = this.shaftsTarget as Target;
    gl.bindFramebuffer(gl.FRAMEBUFFER, shafts.framebuffer);
    gl.viewport(0, 0, shafts.width, shafts.height);
    gl.activeTexture(gl.TEXTURE0 + UNIT_A);
    gl.bindTexture(gl.TEXTURE_2D, this.resolveDepth);
    this.shafts.use().mat4("uInvViewProj", invViewProj);
    drawFullscreen(gl);

    // Dual-filter bloom: walk down the pyramid, then blur back up, adding each level.
    this.down.use();
    let source: WebGLTexture = this.resolveColor as WebGLTexture;
    let sourceWidth = width;
    let sourceHeight = height;
    this.bloomDown.forEach((target, level) => {
      gl.bindFramebuffer(gl.FRAMEBUFFER, target.framebuffer);
      gl.viewport(0, 0, target.width, target.height);
      gl.bindTexture(gl.TEXTURE_2D, source);
      this.down
        .vec2("uTexel", 1 / sourceWidth, 1 / sourceHeight)
        .float("uThreshold", level === 0 ? 1.5 : 0);
      drawFullscreen(gl);
      source = target.texture;
      sourceWidth = target.width;
      sourceHeight = target.height;
    });
    this.up.use();
    gl.blendFunc(gl.ONE, gl.ONE);
    for (let level = BLOOM_LEVELS - 2; level >= 0; level--) {
      const target = this.bloomUp[level];
      gl.bindFramebuffer(gl.FRAMEBUFFER, target.framebuffer);
      gl.viewport(0, 0, target.width, target.height);
      // Start from this level's own downsample, then add the blurred level below.
      gl.disable(gl.BLEND);
      gl.bindTexture(gl.TEXTURE_2D, this.bloomDown[level].texture);
      this.up.vec2("uTexel", 1 / target.width, 1 / target.height);
      drawFullscreen(gl);
      gl.enable(gl.BLEND);
      const below =
        level === BLOOM_LEVELS - 2 ? this.bloomDown[level + 1] : this.bloomUp[level + 1];
      gl.bindTexture(gl.TEXTURE_2D, below.texture);
      this.up.vec2("uTexel", 1 / below.width, 1 / below.height);
      drawFullscreen(gl);
    }
    gl.disable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
    gl.activeTexture(gl.TEXTURE0 + UNIT_A);
    gl.bindTexture(gl.TEXTURE_2D, this.resolveColor);
    gl.activeTexture(gl.TEXTURE0 + UNIT_B);
    gl.bindTexture(gl.TEXTURE_2D, shafts.texture);
    gl.activeTexture(gl.TEXTURE0 + UNIT_C);
    gl.bindTexture(gl.TEXTURE_2D, this.bloomUp[0].texture);
    this.composite.use().float("uTime", time);
    drawFullscreen(gl);
    gl.activeTexture(gl.TEXTURE0 + UNIT_A);
  }
}

/** The directional-light depth map. Its texture stays bound to UNIT_SHADOW. */
export class ShadowMap {
  private readonly gl: WebGL2RenderingContext;
  private readonly framebuffer: WebGLFramebuffer;
  private readonly size = 1024;

  constructor(gl: WebGL2RenderingContext) {
    this.gl = gl;
    const texture = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0 + UNIT_SHADOW);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.DEPTH_COMPONENT24, this.size, this.size);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_COMPARE_MODE, gl.COMPARE_REF_TO_TEXTURE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_COMPARE_FUNC, gl.LEQUAL);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    this.framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.TEXTURE_2D, texture, 0);
    gl.drawBuffers([gl.NONE]);
    gl.readBuffer(gl.NONE);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  begin(): void {
    const { gl } = this;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);
    gl.viewport(0, 0, this.size, this.size);
    gl.clear(gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
  }
}
