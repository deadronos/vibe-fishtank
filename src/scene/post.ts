/** The frame pipeline's screen-space half: MSAA resolve, light shafts, bloom and tone mapping. */

import type { Device, Pass, Pipeline, Texture, TextureDesc, Uniforms } from "../lib/gpu/types";
import type { Floats } from "../lib/math";
import {
  FRAME_GLSL,
  FRAME_WGSL,
  FULLSCREEN_GLSL,
  FULLSCREEN_WGSL,
  PRELUDE_WGSL,
  shader,
  WATER_GLSL,
  WATER_TEXTURES,
  WATER_WGSL,
  type SceneTargets,
} from "./common";

const BLOOM_LEVELS = 5;
const SHADOW_SIZE = 1024;

const SHAFTS_FS = `${FRAME_GLSL}${WATER_GLSL}
in vec2 vUv;
uniform sampler2D uDepth;
layout(std140) uniform Params { mat4 uInvViewProj; };
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

/**
 * WebGPU can't resolve depth, so this reads the multisampled depth buffer's first sample. The
 * WebGPU device always renders the scene with 4x MSAA, so the binding is always multisampled.
 */
const SHAFTS_WGSL = `${FRAME_WGSL}${WATER_WGSL}${FULLSCREEN_WGSL}
@group(0) @binding(5) var depthTex: texture_depth_multisampled_2d;
@group(1) @binding(0) var<uniform> invViewProj: mat4x4f;

fn ign(p: vec2f) -> f32 {
  return fract(52.9829189 * fract(dot(p, vec2f(0.06711056, 0.00583715))));
}

@fragment fn fs(v: FsOut) -> @location(0) vec4f {
  let ndc = vec2f(v.uv.x * 2.0 - 1.0, 1.0 - v.uv.y * 2.0);
  let farPoint = invViewProj * vec4f(ndc, 1.0, 1.0);
  let rd = normalize(farPoint.xyz / farPoint.w - frame.camPos);
  let hit = boxRay(frame.camPos, rd);
  let a = max(hit.x, 0.0);
  var b = hit.y;
  if (b <= a) { return vec4f(0.0); }
  let size = vec2f(textureDimensions(depthTex));
  let texel = vec2i(clamp(v.uv * size, vec2f(0.0), size - 1.0));
  let sceneDepth = textureLoad(depthTex, texel, 0);
  let scene = invViewProj * vec4f(ndc, sceneDepth * 2.0 - 1.0, 1.0);
  b = min(b, length(scene.xyz / scene.w - frame.camPos));
  if (b <= a) { return vec4f(0.0); }
  const STEPS = 14;
  let stepLength = (b - a) / f32(STEPS);
  var t = a + stepLength * ign(v.pos.xy);
  var sum = vec3f(0.0);
  for (var i = 0; i < STEPS; i++) {
    let p = frame.camPos + rd * t;
    let depth = max(WATER_Y - p.y, 0.0);
    let uv = lightUv(p);
    var beam = textureSampleLevel(causticsTex, causticsSamp, uv * 0.42, 3.4).r
      + 0.5 * textureSampleLevel(causticsTex, causticsSamp, uv * 0.19 + 0.3, 3.8).r;
    beam = smoothstep(0.08, 0.34, beam);
    let lp = shadowCoord(p);
    let visible = textureSampleCompareLevel(shadowTex, shadowSamp, lp.xy, lp.z - 0.004);
    sum += beam * visible * exp(-depth * ABSORB) * exp(-(t - a) * FOG);
    t += stepLength;
  }
  return vec4f(sum * stepLength * frame.lightCol * 0.05, 1.0);
}`;

/** Bindings for post passes that read one filtered texture. */
const SOURCE_WGSL = `
@group(0) @binding(1) var tex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(1) @binding(0) var<uniform> p: vec4f;
fn at(uv: vec2f) -> vec3f { return textureSample(tex, samp, uv).rgb; }
`;

const DOWN_FS = `
in vec2 vUv;
uniform sampler2D uTex;
layout(std140) uniform Params { vec4 uP; };
out vec4 outColor;
void main() {
  vec2 uTexel = uP.xy;
  float uThreshold = uP.z;
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

const DOWN_WGSL = `${PRELUDE_WGSL}${FULLSCREEN_WGSL}${SOURCE_WGSL}
@fragment fn fs(v: FsOut) -> @location(0) vec4f {
  let texel = p.xy;
  let threshold = p.z;
  var c = at(v.uv) * 0.25;
  c += at(v.uv + texel * vec2f(-1.0, -1.0)) * 0.1875;
  c += at(v.uv + texel * vec2f(1.0, -1.0)) * 0.1875;
  c += at(v.uv + texel * vec2f(-1.0, 1.0)) * 0.1875;
  c += at(v.uv + texel * vec2f(1.0, 1.0)) * 0.1875;
  if (threshold > 0.0) {
    let l = max(c.r, max(c.g, c.b));
    c *= smoothstep(threshold * 0.6, threshold * 1.4, l);
    c = min(c, vec3f(8.0));
  }
  return vec4f(c, 1.0);
}`;

const UP_FS = `
in vec2 vUv;
uniform sampler2D uTex;
layout(std140) uniform Params { vec4 uP; };
out vec4 outColor;
void main() {
  vec2 uTexel = uP.xy;
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

const UP_WGSL = `${PRELUDE_WGSL}${FULLSCREEN_WGSL}${SOURCE_WGSL}
@fragment fn fs(v: FsOut) -> @location(0) vec4f {
  let texel = p.xy;
  var c = at(v.uv) * 4.0;
  c += at(v.uv + texel * vec2f(-1.0, 0.0)) * 2.0;
  c += at(v.uv + texel * vec2f(1.0, 0.0)) * 2.0;
  c += at(v.uv + texel * vec2f(0.0, -1.0)) * 2.0;
  c += at(v.uv + texel * vec2f(0.0, 1.0)) * 2.0;
  c += at(v.uv + texel * vec2f(-1.0, -1.0));
  c += at(v.uv + texel * vec2f(1.0, -1.0));
  c += at(v.uv + texel * vec2f(-1.0, 1.0));
  c += at(v.uv + texel * vec2f(1.0, 1.0));
  return vec4f(c / 16.0, 1.0);
}`;

const COMPOSITE_FS = `
in vec2 vUv;
uniform sampler2D uScene;
uniform sampler2D uShafts;
uniform sampler2D uBloom;
layout(std140) uniform Params { vec4 uP; };
out vec4 outColor;

vec3 aces(vec3 x) {
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}
float noise(vec2 p) {
  return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}
void main() {
  float uTime = uP.x;
  vec3 hdr = texture(uScene, vUv).rgb + texture(uShafts, vUv).rgb + texture(uBloom, vUv).rgb * 0.2;
  vec2 q = vUv - 0.5;
  hdr *= 1.0 - dot(q, q) * 0.72;
  vec3 c = aces(hdr * 1.08);
  c = pow(c, vec3(1.0 / 2.2));
  c += (noise(gl_FragCoord.xy + fract(uTime) * 61.0) - 0.5) / 255.0 * 1.6;
  outColor = vec4(c, 1.0);
}`;

const COMPOSITE_WGSL = `${PRELUDE_WGSL}${FULLSCREEN_WGSL}
@group(0) @binding(1) var sceneTex: texture_2d<f32>;
@group(0) @binding(2) var sceneSamp: sampler;
@group(0) @binding(3) var shaftsTex: texture_2d<f32>;
@group(0) @binding(4) var shaftsSamp: sampler;
@group(0) @binding(5) var bloomTex: texture_2d<f32>;
@group(0) @binding(6) var bloomSamp: sampler;
@group(1) @binding(0) var<uniform> p: vec4f;

fn aces(x: vec3f) -> vec3f {
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3f(0.0), vec3f(1.0));
}
fn noise(q: vec2f) -> f32 {
  return fract(sin(dot(q, vec2f(12.9898, 78.233))) * 43758.5453);
}
@fragment fn fs(v: FsOut) -> @location(0) vec4f {
  var hdr = textureSample(sceneTex, sceneSamp, v.uv).rgb
    + textureSample(shaftsTex, shaftsSamp, v.uv).rgb
    + textureSample(bloomTex, bloomSamp, v.uv).rgb * 0.2;
  let q = v.uv - 0.5;
  hdr *= 1.0 - dot(q, q) * 0.72;
  var c = aces(hdr * 1.08);
  c = pow(c, vec3f(1.0 / 2.2));
  c += (noise(v.pos.xy + fract(p.x) * 61.0) - 0.5) / 255.0 * 1.6;
  return vec4f(c, 1.0);
}`;

interface Targets {
  sceneColor: Texture;
  sceneDepth: Texture;
  resolveColor: Texture;
  /** Only on devices that can resolve depth; otherwise shafts read `sceneDepth` directly. */
  resolveDepth: Texture | null;
  shafts: Texture;
  bloomDown: Texture[];
  bloomUp: Texture[];
}

export class PostPipeline {
  readonly targets: SceneTargets;
  private readonly device: Device;
  private readonly shafts: Pipeline;
  private readonly down: Pipeline;
  private readonly up: Pipeline;
  private readonly upAdd: Pipeline;
  private readonly composite: Pipeline;
  private readonly vec4: Floats = new Float32Array(4);
  private textures: Targets | null = null;
  private width = 0;
  private height = 0;

  constructor(device: Device) {
    this.device = device;
    const { caps } = device;
    this.targets = { color: caps.hdrFormat, samples: caps.samples };
    const color = caps.hdrFormat;
    const fullscreen = { vertex: "none", params: 16, color } as const;
    const source = [{ name: "uTex", kind: "float" }] as const;

    this.shafts = device.createPipeline({
      ...fullscreen,
      label: "light shafts",
      shader: shader(FULLSCREEN_GLSL, SHAFTS_FS, SHAFTS_WGSL),
      frame: true,
      params: 64,
      textures: [
        ...WATER_TEXTURES,
        { name: "uDepth", kind: "depth", multisampled: !caps.depthResolve },
      ],
    });
    this.down = device.createPipeline({
      ...fullscreen,
      label: "bloom down",
      shader: shader(FULLSCREEN_GLSL, DOWN_FS, DOWN_WGSL),
      textures: source,
    });
    const up = shader(FULLSCREEN_GLSL, UP_FS, UP_WGSL);
    this.up = device.createPipeline({
      ...fullscreen,
      label: "bloom up",
      shader: up,
      textures: source,
    });
    this.upAdd = device.createPipeline({
      ...fullscreen,
      label: "bloom up add",
      shader: up,
      textures: source,
      blend: "additive",
    });
    this.composite = device.createPipeline({
      ...fullscreen,
      label: "composite",
      shader: shader(FULLSCREEN_GLSL, COMPOSITE_FS, COMPOSITE_WGSL),
      color: "canvas",
      textures: [
        { name: "uScene", kind: "float" },
        { name: "uShafts", kind: "float" },
        { name: "uBloom", kind: "float" },
      ],
    });
  }

  resize(width: number, height: number): void {
    if (width === this.width && height === this.height) return;
    this.width = width;
    this.height = height;
    const { device, targets } = this;
    if (this.textures) {
      const old = this.textures;
      for (const texture of [
        old.sceneColor,
        old.sceneDepth,
        old.resolveColor,
        old.resolveDepth,
        old.shafts,
        ...old.bloomDown,
        ...old.bloomUp,
      ]) {
        texture?.destroy();
      }
    }
    const make = (label: string, w: number, h: number, extra: Partial<TextureDesc> = {}): Texture =>
      device.createTexture({ label, width: w, height: h, format: targets.color, ...extra });

    const bloomDown: Texture[] = [];
    const bloomUp: Texture[] = [];
    let w = width;
    let h = height;
    for (let i = 0; i < BLOOM_LEVELS; i++) {
      w = Math.max(2, w >> 1);
      h = Math.max(2, h >> 1);
      bloomDown.push(make(`bloom down ${i}`, w, h));
      if (i < BLOOM_LEVELS - 1) bloomUp.push(make(`bloom up ${i}`, w, h));
    }
    this.textures = {
      sceneColor: make("scene", width, height, { samples: targets.samples }),
      sceneDepth: make("scene depth", width, height, {
        format: "depth24",
        samples: targets.samples,
        filter: "nearest",
      }),
      resolveColor: make("scene resolved", width, height),
      resolveDepth: device.caps.depthResolve
        ? make("scene depth resolved", width, height, { format: "depth24", filter: "nearest" })
        : null,
      shafts: make("shafts", Math.max(1, width >> 1), Math.max(1, height >> 1)),
      bloomDown,
      bloomUp,
    };
  }

  /** Starts the HDR scene pass, which resolves when it ends. */
  beginScene(): Pass {
    const t = this.current();
    return this.device.beginPass({
      label: "scene",
      color: { target: t.sceneColor, resolve: t.resolveColor, clear: [0, 0, 0, 1] },
      depth: { target: t.sceneDepth, resolve: t.resolveDepth ?? undefined, clear: 1 },
    });
  }

  /** Computes shafts and bloom, and tone maps to the canvas. */
  finish(frame: Uniforms, water: readonly Texture[], invViewProj: Floats, time: number): void {
    const { device, vec4 } = this;
    const t = this.current();
    const fullscreen = (
      label: string,
      target: Texture | "canvas",
      draw: (pass: Pass) => void,
    ): void => {
      const pass = device.beginPass({ label, color: { target, clear: [0, 0, 0, 1] } });
      draw(pass);
      pass.end();
    };
    const params = (x: number, y: number, z = 0, w = 0): Uniforms => {
      vec4.set([x, y, z, w]);
      return device.uniforms(vec4);
    };

    fullscreen("light shafts", t.shafts, (pass) => {
      pass.draw(this.shafts, null, {
        frame,
        params: device.uniforms(invViewProj),
        textures: [...water, t.resolveDepth ?? t.sceneDepth],
      });
    });

    // Dual-filter bloom: walk down the pyramid, then blur back up, adding each level.
    let source = t.resolveColor;
    t.bloomDown.forEach((target, level) => {
      fullscreen(`bloom down ${level}`, target, (pass) => {
        pass.draw(this.down, null, {
          params: params(1 / source.width, 1 / source.height, level === 0 ? 1.5 : 0),
          textures: [source],
        });
      });
      source = target;
    });
    for (let level = BLOOM_LEVELS - 2; level >= 0; level--) {
      const target = t.bloomUp[level];
      const below = level === BLOOM_LEVELS - 2 ? t.bloomDown[level + 1] : t.bloomUp[level + 1];
      fullscreen(`bloom up ${level}`, target, (pass) => {
        // Start from this level's own downsample, then add the blurred level below.
        pass.draw(this.up, null, {
          params: params(1 / target.width, 1 / target.height),
          textures: [t.bloomDown[level]],
        });
        pass.draw(this.upAdd, null, {
          params: params(1 / below.width, 1 / below.height),
          textures: [below],
        });
      });
    }

    fullscreen("composite", "canvas", (pass) => {
      pass.draw(this.composite, null, {
        params: params(time, 0),
        textures: [t.resolveColor, t.shafts, t.bloomUp[0]],
      });
    });
  }

  private current(): Targets {
    if (!this.textures) throw new Error("PostPipeline.resize wasn't called");
    return this.textures;
  }
}

/** The directional light's depth map. */
export class ShadowMap {
  readonly texture: Texture;
  private readonly device: Device;

  constructor(device: Device) {
    this.device = device;
    this.texture = device.createTexture({
      label: "shadow map",
      width: SHADOW_SIZE,
      height: SHADOW_SIZE,
      format: "depth24",
      compare: true,
    });
  }

  begin(): Pass {
    return this.device.beginPass({ label: "shadow", depth: { target: this.texture, clear: 1 } });
  }
}
