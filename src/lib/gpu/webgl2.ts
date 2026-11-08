/** The WebGL2 backend. Every call runs immediately; passes are just framebuffer bindings. */

import type { Floats } from "../math";
import {
  ATTR,
  type Capabilities,
  type Device,
  type DrawBindings,
  type Geometry,
  type Mesh,
  type Pass,
  type PassDesc,
  type Pipeline,
  type PipelineDesc,
  type Texture,
  type TextureDesc,
  type Uniforms,
} from "./types";

const HEADER = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2DShadow;
`;

const DEPTH_ONLY_FS = "void main() {}";
const FRAME_BINDING = 0;
const PARAMS_BINDING = 1;
/** Uniform memory is a ring, so a frame's writes rarely touch data the GPU is still reading. */
const RING_BYTES = 256 * 1024;

let nextId = 1;

class GLTexture implements Texture {
  readonly id = nextId++;
  readonly desc: TextureDesc;
  readonly texture: WebGLTexture | null = null;
  readonly renderbuffer: WebGLRenderbuffer | null = null;
  private readonly device: WebGL2Device;

  constructor(device: WebGL2Device, desc: TextureDesc) {
    this.device = device;
    this.desc = desc;
    const { gl } = device;
    const internal = {
      rgba8: gl.RGBA8,
      rgba16f: gl.RGBA16F,
      depth24: gl.DEPTH_COMPONENT24,
    }[desc.format];
    const samples = desc.samples ?? 1;
    if (samples > 1) {
      // Multisampled targets are never sampled in WebGL2: they're resolved with a blit.
      this.renderbuffer = gl.createRenderbuffer();
      gl.bindRenderbuffer(gl.RENDERBUFFER, this.renderbuffer);
      gl.renderbufferStorageMultisample(
        gl.RENDERBUFFER,
        samples,
        internal,
        desc.width,
        desc.height,
      );
      return;
    }
    this.texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.texture);
    const levels = desc.mipmaps ? Math.floor(Math.log2(Math.max(desc.width, desc.height))) + 1 : 1;
    gl.texStorage2D(gl.TEXTURE_2D, levels, internal, desc.width, desc.height);
    const wrap = desc.wrap === "repeat" ? gl.REPEAT : gl.CLAMP_TO_EDGE;
    const linear = desc.filter !== "nearest";
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, wrap);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, wrap);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, linear ? gl.LINEAR : gl.NEAREST);
    gl.texParameteri(
      gl.TEXTURE_2D,
      gl.TEXTURE_MIN_FILTER,
      !linear ? gl.NEAREST : desc.mipmaps ? gl.LINEAR_MIPMAP_LINEAR : gl.LINEAR,
    );
    if (desc.compare) {
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_COMPARE_MODE, gl.COMPARE_REF_TO_TEXTURE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_COMPARE_FUNC, gl.LEQUAL);
    }
    if (desc.anisotropic && device.anisotropic) {
      gl.texParameterf(
        gl.TEXTURE_2D,
        device.anisotropic.TEXTURE_MAX_ANISOTROPY_EXT,
        device.caps.anisotropy,
      );
    }
  }

  get width(): number {
    return this.desc.width;
  }

  get height(): number {
    return this.desc.height;
  }

  destroy(): void {
    this.device.forget(this);
    this.device.gl.deleteTexture(this.texture);
    this.device.gl.deleteRenderbuffer(this.renderbuffer);
  }
}

class GLPipeline implements Pipeline {
  readonly desc: PipelineDesc;
  readonly program: WebGLProgram;

  constructor(gl: WebGL2RenderingContext, desc: PipelineDesc) {
    this.desc = desc;
    const fragment = desc.color ? desc.shader.glsl.fragment : DEPTH_ONLY_FS;
    const program = gl.createProgram();
    gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, desc.shader.glsl.vertex, desc.label));
    gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, fragment, desc.label));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS) && !gl.isContextLost()) {
      throw new Error(`${desc.label}: link failed\n${gl.getProgramInfoLog(program)}`);
    }
    this.program = program;
    const bind = (name: string, binding: number): void => {
      const block = gl.getUniformBlockIndex(program, name);
      if (block !== gl.INVALID_INDEX) gl.uniformBlockBinding(program, block, binding);
    };
    bind("Frame", FRAME_BINDING);
    bind("Params", PARAMS_BINDING);
    gl.useProgram(program);
    desc.textures?.forEach((texture, unit) => {
      gl.uniform1i(gl.getUniformLocation(program, texture.name), unit);
    });
  }
}

function compile(
  gl: WebGL2RenderingContext,
  type: number,
  source: string,
  label: string,
): WebGLShader {
  const shader = gl.createShader(type);
  if (!shader) throw new Error("Could not create shader");
  gl.shaderSource(shader, HEADER + source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS) && !gl.isContextLost()) {
    const numbered = (HEADER + source)
      .split("\n")
      .map((line, i) => `${String(i + 1).padStart(4)} ${line}`)
      .join("\n");
    throw new Error(`${label}: compile failed\n${gl.getShaderInfoLog(shader)}\n${numbered}`);
  }
  return shader;
}

class GLMesh implements Mesh {
  readonly vao: WebGLVertexArrayObject;
  readonly count: number;
  instanceCount = -1;
  private readonly gl: WebGL2RenderingContext;
  private readonly instanceBuffer: WebGLBuffer | null = null;
  private readonly instanceFloats: number;

  constructor(gl: WebGL2RenderingContext, geometry: Geometry, instances?: readonly number[]) {
    this.gl = gl;
    this.instanceFloats = instances?.reduce((sum, size) => sum + size, 0) ?? 0;
    this.vao = gl.createVertexArray();
    this.count = geometry.indices.length;
    gl.bindVertexArray(this.vao);
    const attribute = (slot: number, data: Floats, size: number): void => {
      gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
      gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(slot);
      gl.vertexAttribPointer(slot, size, gl.FLOAT, false, 0, 0);
    };
    attribute(ATTR.position, geometry.positions, 3);
    attribute(ATTR.normal, geometry.normals, 3);
    attribute(ATTR.uv, geometry.uvs, 2);
    attribute(ATTR.extra, geometry.extras, 1);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, geometry.indices, gl.STATIC_DRAW);

    if (instances) {
      this.instanceCount = 0;
      this.instanceBuffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceBuffer);
      const stride = this.instanceFloats * 4;
      let offset = 0;
      instances.forEach((size, i) => {
        const slot = ATTR.instance + i;
        gl.enableVertexAttribArray(slot);
        gl.vertexAttribPointer(slot, size, gl.FLOAT, false, stride, offset);
        gl.vertexAttribDivisor(slot, 1);
        offset += size * 4;
      });
    }
    gl.bindVertexArray(null);
  }

  setInstances(data: Floats, count: number): void {
    const { gl } = this;
    this.instanceCount = count;
    if (count === 0) return;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, data.subarray(0, count * this.instanceFloats), gl.DYNAMIC_DRAW);
  }

  draw(): void {
    const { gl } = this;
    if (this.instanceCount === 0) return;
    gl.bindVertexArray(this.vao);
    if (this.instanceCount > 0) {
      gl.drawElementsInstanced(gl.TRIANGLES, this.count, gl.UNSIGNED_INT, 0, this.instanceCount);
    } else {
      gl.drawElements(gl.TRIANGLES, this.count, gl.UNSIGNED_INT, 0);
    }
  }
}

class GLPass implements Pass {
  private readonly device: WebGL2Device;
  private readonly desc: PassDesc;
  private readonly framebuffer: WebGLFramebuffer | null;

  constructor(device: WebGL2Device, desc: PassDesc) {
    this.device = device;
    this.desc = desc;
    const { gl } = device;
    const color = desc.color?.target;
    const level = desc.color?.level ?? 0;
    this.framebuffer =
      color === "canvas"
        ? null
        : device.framebuffer(
            color as GLTexture | undefined,
            level,
            desc.depth?.target as GLTexture,
          );
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);
    if (color === "canvas") {
      gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
    } else {
      const size = color ?? (desc.depth?.target as Texture);
      gl.viewport(0, 0, Math.max(1, size.width >> level), Math.max(1, size.height >> level));
    }
    let mask = 0;
    const clearColor = desc.color?.clear;
    if (clearColor) {
      gl.clearColor(clearColor[0], clearColor[1], clearColor[2], clearColor[3]);
      mask |= gl.COLOR_BUFFER_BIT;
    }
    if (desc.depth?.clear !== undefined) {
      gl.depthMask(true);
      gl.clearDepth(desc.depth.clear);
      mask |= gl.DEPTH_BUFFER_BIT;
    }
    if (mask) gl.clear(mask);
  }

  draw(pipeline: Pipeline, mesh: Mesh | null, bindings: DrawBindings = {}): void {
    const { gl, ring } = this.device;
    const { desc, program } = pipeline as GLPipeline;
    gl.useProgram(program);
    if (desc.depth) gl.enable(gl.DEPTH_TEST);
    else gl.disable(gl.DEPTH_TEST);
    gl.depthMask(desc.depth === "write");
    const blend = desc.blend ?? "none";
    if (blend === "none") {
      gl.disable(gl.BLEND);
    } else {
      gl.enable(gl.BLEND);
      if (blend === "alpha") gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      else if (blend === "premultiplied") gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
      else gl.blendFunc(gl.ONE, gl.ONE);
    }
    if (desc.alphaToCoverage) gl.enable(gl.SAMPLE_ALPHA_TO_COVERAGE);
    else gl.disable(gl.SAMPLE_ALPHA_TO_COVERAGE);

    const { frame, params, textures } = bindings;
    if (frame) gl.bindBufferRange(gl.UNIFORM_BUFFER, FRAME_BINDING, ring, frame.offset, frame.size);
    if (params) {
      gl.bindBufferRange(gl.UNIFORM_BUFFER, PARAMS_BINDING, ring, params.offset, params.size);
    }
    textures?.forEach((texture, unit) => {
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, (texture as GLTexture).texture);
    });

    if (mesh) {
      (mesh as GLMesh).draw();
    } else {
      gl.bindVertexArray(null);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }
  }

  end(): void {
    const { gl } = this.device;
    const { color, depth } = this.desc;
    const resolveColor = color?.resolve as GLTexture | undefined;
    const resolveDepth = depth?.resolve as GLTexture | undefined;
    if (!resolveColor && !resolveDepth) return;
    const target = this.device.framebuffer(resolveColor, 0, resolveDepth);
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, this.framebuffer);
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, target);
    const { width, height } = (resolveColor ?? resolveDepth) as GLTexture;
    const mask =
      (resolveColor ? gl.COLOR_BUFFER_BIT : 0) | (resolveDepth ? gl.DEPTH_BUFFER_BIT : 0);
    gl.blitFramebuffer(0, 0, width, height, 0, 0, width, height, mask, gl.NEAREST);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }
}

export class WebGL2Device implements Device {
  readonly gl: WebGL2RenderingContext;
  readonly caps: Capabilities;
  readonly ring: WebGLBuffer;
  readonly anisotropic: EXT_texture_filter_anisotropic | null;
  private readonly canvas: HTMLCanvasElement;
  private readonly align: number;
  private readonly framebuffers = new Map<string, { fbo: WebGLFramebuffer; ids: number[] }>();
  private ringOffset = 0;
  private lost = false;

  constructor(canvas: HTMLCanvasElement, gl: WebGL2RenderingContext) {
    this.canvas = canvas;
    this.gl = gl;
    gl.disable(gl.CULL_FACE);

    const features = gl.getSupportedExtensions() ?? [];
    const used: string[] = [];
    const float = gl.getExtension("EXT_color_buffer_float") !== null;
    if (float) used.push("EXT_color_buffer_float");
    this.anisotropic = gl.getExtension("EXT_texture_filter_anisotropic");
    let anisotropy = 1;
    if (this.anisotropic) {
      used.push("EXT_texture_filter_anisotropic");
      const max = gl.getParameter(this.anisotropic.MAX_TEXTURE_MAX_ANISOTROPY_EXT) as number;
      anisotropy = Math.min(8, max);
    }
    const supported = float
      ? (gl.getInternalformatParameter(gl.RENDERBUFFER, gl.RGBA16F, gl.SAMPLES) as Int32Array)[0]
      : (gl.getParameter(gl.MAX_SAMPLES) as number);
    this.caps = {
      backend: "webgl2",
      adapter: rendererName(gl),
      hdrFormat: float ? "rgba16f" : "rgba8",
      samples: Math.max(1, Math.min(4, supported ?? 0)),
      depthResolve: true,
      anisotropy,
      features,
      used,
    };

    this.align = gl.getParameter(gl.UNIFORM_BUFFER_OFFSET_ALIGNMENT) as number;
    this.ring = gl.createBuffer();
    gl.bindBuffer(gl.UNIFORM_BUFFER, this.ring);
    gl.bufferData(gl.UNIFORM_BUFFER, RING_BYTES, gl.DYNAMIC_DRAW);
  }

  createMesh(geometry: Geometry, instances?: readonly number[]): Mesh {
    return new GLMesh(this.gl, geometry, instances);
  }

  createTexture(desc: TextureDesc): Texture {
    return new GLTexture(this, desc);
  }

  createPipeline(desc: PipelineDesc): Pipeline {
    return new GLPipeline(this.gl, desc);
  }

  beginFrame(): void {}

  uniforms(data: Floats): Uniforms {
    const { gl } = this;
    const size = data.byteLength;
    let offset = Math.ceil(this.ringOffset / this.align) * this.align;
    if (offset + size > RING_BYTES) offset = 0;
    gl.bindBuffer(gl.UNIFORM_BUFFER, this.ring);
    gl.bufferSubData(gl.UNIFORM_BUFFER, offset, data);
    this.ringOffset = offset + size;
    return { offset, size };
  }

  beginPass(desc: PassDesc): Pass {
    return new GLPass(this, desc);
  }

  generateMipmaps(texture: Texture): void {
    const { gl } = this;
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, (texture as GLTexture).texture);
    gl.generateMipmap(gl.TEXTURE_2D);
  }

  endFrame(): void {}

  validate(): Promise<void> {
    // Shaders compile synchronously, so any error has already been thrown.
    return Promise.resolve();
  }

  onLost(callback: (message: string) => void): void {
    this.canvas.addEventListener("webglcontextlost", (event) => {
      event.preventDefault();
      if (!this.lost) callback("The WebGL2 context was lost.");
      this.lost = true;
    });
  }

  destroy(): void {
    this.lost = true;
    this.gl.getExtension("WEBGL_lose_context")?.loseContext();
  }

  /** A cached framebuffer for this combination of attachments. */
  framebuffer(
    color: GLTexture | undefined,
    level: number,
    depth: GLTexture | undefined,
  ): WebGLFramebuffer {
    const key = `${color?.id ?? 0}:${level}:${depth?.id ?? 0}`;
    const cached = this.framebuffers.get(key);
    if (cached) return cached.fbo;
    const { gl } = this;
    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    const attach = (attachment: number, texture: GLTexture, mip: number): void => {
      if (texture.renderbuffer) {
        gl.framebufferRenderbuffer(
          gl.FRAMEBUFFER,
          attachment,
          gl.RENDERBUFFER,
          texture.renderbuffer,
        );
      } else {
        gl.framebufferTexture2D(gl.FRAMEBUFFER, attachment, gl.TEXTURE_2D, texture.texture, mip);
      }
    };
    if (color) {
      attach(gl.COLOR_ATTACHMENT0, color, level);
    } else {
      gl.drawBuffers([gl.NONE]);
      gl.readBuffer(gl.NONE);
    }
    if (depth) attach(gl.DEPTH_ATTACHMENT, depth, 0);
    this.framebuffers.set(key, { fbo, ids: [color?.id ?? 0, depth?.id ?? 0] });
    return fbo;
  }

  /** Drops framebuffers that use a texture about to be deleted. */
  forget(texture: GLTexture): void {
    for (const [key, entry] of this.framebuffers) {
      if (entry.ids.includes(texture.id)) {
        this.gl.deleteFramebuffer(entry.fbo);
        this.framebuffers.delete(key);
      }
    }
  }
}

function rendererName(gl: WebGL2RenderingContext): string {
  let name = gl.getParameter(gl.RENDERER) as string;
  // Chrome masks RENDERER, but still answers through the debug extension.
  if (!name || name.startsWith("WebKit")) {
    const debug = gl.getExtension("WEBGL_debug_renderer_info");
    if (debug) name = gl.getParameter(debug.UNMASKED_RENDERER_WEBGL) as string;
  }
  return name ? shortRendererName(name) : "unknown";
}

/**
 * ANGLE wraps the GPU's name, as in "ANGLE (Apple, ANGLE Metal Renderer: Apple M4, Unspecified
 * Version)" or "ANGLE (NVIDIA, NVIDIA GeForce RTX 3080 Direct3D11 vs_5_0 ps_5_0, D3D11)".
 */
export function shortRendererName(name: string): string {
  const inner = /^ANGLE \([^,]+, ([^,]+)/.exec(name)?.[1];
  if (!inner) return name;
  return inner
    .replace(/^.*Renderer: /, "")
    .replace(/ (Direct3D|vs_|ps_).*$/, "")
    .trim();
}

export function createWebGL2Device(canvas: HTMLCanvasElement): WebGL2Device | null {
  const gl = canvas.getContext("webgl2", {
    antialias: false,
    alpha: false,
    depth: false,
    stencil: false,
    powerPreference: "high-performance",
  });
  return gl ? new WebGL2Device(canvas, gl) : null;
}
