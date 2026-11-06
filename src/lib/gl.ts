/** Thin wrappers around WebGL2: programs, meshes, instancing and render targets. */

import type { Floats } from "./math";

const HEADER = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2DShadow;
`;

/** Fixed attribute slots, so every mesh works with every shader. */
export const ATTR = { position: 0, normal: 1, uv: 2, extra: 3, instance: 4 } as const;

/** Uniform block binding used by the shared per-frame data. */
export const FRAME_BINDING = 0;

function compile(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader {
  const shader = gl.createShader(type);
  if (!shader) throw new Error("Could not create shader");
  gl.shaderSource(shader, HEADER + source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS) && !gl.isContextLost()) {
    const numbered = (HEADER + source)
      .split("\n")
      .map((line, i) => `${String(i + 1).padStart(4)} ${line}`)
      .join("\n");
    throw new Error(`Shader compile failed:\n${gl.getShaderInfoLog(shader)}\n${numbered}`);
  }
  return shader;
}

export class Program {
  private readonly locations = new Map<string, WebGLUniformLocation | null>();
  private readonly handle: WebGLProgram;
  private readonly gl: WebGL2RenderingContext;

  constructor(gl: WebGL2RenderingContext, vertex: string, fragment: string) {
    this.gl = gl;
    const program = gl.createProgram();
    gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, vertex));
    gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, fragment));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS) && !gl.isContextLost()) {
      throw new Error(`Program link failed:\n${gl.getProgramInfoLog(program)}`);
    }
    this.handle = program;
    const block = gl.getUniformBlockIndex(program, "Frame");
    if (block !== gl.INVALID_INDEX) gl.uniformBlockBinding(program, block, FRAME_BINDING);
  }

  use(): this {
    this.gl.useProgram(this.handle);
    return this;
  }

  private loc(name: string): WebGLUniformLocation | null {
    let location = this.locations.get(name);
    if (location === undefined) {
      location = this.gl.getUniformLocation(this.handle, name);
      this.locations.set(name, location);
    }
    return location;
  }

  int(name: string, value: number): this {
    this.gl.uniform1i(this.loc(name), value);
    return this;
  }

  float(name: string, value: number): this {
    this.gl.uniform1f(this.loc(name), value);
    return this;
  }

  vec2(name: string, x: number, y: number): this {
    this.gl.uniform2f(this.loc(name), x, y);
    return this;
  }

  vec3(name: string, x: number, y: number, z: number): this {
    this.gl.uniform3f(this.loc(name), x, y, z);
    return this;
  }

  mat4(name: string, value: Floats): this {
    this.gl.uniformMatrix4fv(this.loc(name), false, value);
    return this;
  }
}

export interface Geometry {
  positions: Floats;
  normals: Floats;
  uvs: Floats;
  extras: Floats;
  indices: Uint32Array<ArrayBuffer>;
}

export class Mesh {
  private readonly gl: WebGL2RenderingContext;
  private readonly vao: WebGLVertexArrayObject;
  private readonly count: number;
  private instanceBuffer: WebGLBuffer | null = null;
  private instanceCount = 0;

  constructor(gl: WebGL2RenderingContext, geometry: Geometry) {
    this.gl = gl;
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
    gl.bindVertexArray(null);
  }

  /** Declares per-instance attributes, starting at ATTR.instance. Each size is at most 4. */
  instanced(sizes: readonly number[]): this {
    const { gl } = this;
    gl.bindVertexArray(this.vao);
    this.instanceBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceBuffer);
    const stride = sizes.reduce((sum, size) => sum + size, 0) * 4;
    let offset = 0;
    sizes.forEach((size, i) => {
      const slot = ATTR.instance + i;
      gl.enableVertexAttribArray(slot);
      gl.vertexAttribPointer(slot, size, gl.FLOAT, false, stride, offset);
      gl.vertexAttribDivisor(slot, 1);
      offset += size * 4;
    });
    gl.bindVertexArray(null);
    return this;
  }

  setInstances(data: Floats, count: number, usage: number = this.gl.DYNAMIC_DRAW): void {
    const { gl } = this;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, data, usage);
    this.instanceCount = count;
  }

  draw(): void {
    const { gl } = this;
    gl.bindVertexArray(this.vao);
    if (this.instanceBuffer) {
      gl.drawElementsInstanced(gl.TRIANGLES, this.count, gl.UNSIGNED_INT, 0, this.instanceCount);
    } else {
      gl.drawElements(gl.TRIANGLES, this.count, gl.UNSIGNED_INT, 0);
    }
  }
}

export interface Target {
  framebuffer: WebGLFramebuffer;
  texture: WebGLTexture;
  width: number;
  height: number;
}

export interface ColorFormat {
  internal: number;
  type: number;
}

/** A framebuffer with a single sampled color texture. */
export function createTarget(
  gl: WebGL2RenderingContext,
  width: number,
  height: number,
  format: ColorFormat,
  options: { repeat?: boolean; mipmaps?: boolean } = {},
): Target {
  const texture = gl.createTexture();
  // A spare unit, so creating a target never disturbs a texture bound for the whole frame.
  gl.activeTexture(gl.TEXTURE7);
  gl.bindTexture(gl.TEXTURE_2D, texture);
  const levels = options.mipmaps ? Math.floor(Math.log2(Math.max(width, height))) + 1 : 1;
  gl.texStorage2D(gl.TEXTURE_2D, levels, format.internal, width, height);
  const wrap = options.repeat ? gl.REPEAT : gl.CLAMP_TO_EDGE;
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, wrap);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, wrap);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(
    gl.TEXTURE_2D,
    gl.TEXTURE_MIN_FILTER,
    options.mipmaps ? gl.LINEAR_MIPMAP_LINEAR : gl.LINEAR,
  );
  const framebuffer = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  return { framebuffer, texture, width, height };
}

export function deleteTarget(gl: WebGL2RenderingContext, target: Target): void {
  gl.deleteFramebuffer(target.framebuffer);
  gl.deleteTexture(target.texture);
}

/** Vertex shader for a full-screen triangle; draw with `drawFullscreen`. */
export const FULLSCREEN_VS = `
out vec2 vUv;
void main() {
  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  vUv = p;
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

export function drawFullscreen(gl: WebGL2RenderingContext): void {
  gl.bindVertexArray(null);
  gl.drawArrays(gl.TRIANGLES, 0, 3);
}
