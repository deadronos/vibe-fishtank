/**
 * A small, WebGPU-shaped rendering interface. The scene is written against it once, and two
 * backends implement it: WebGPU, which records each frame into one command encoder, and WebGL2,
 * which runs every call immediately.
 */

import type { Floats } from "../math";

export type BackendKind = "webgpu" | "webgl2";

/** Mesh data with a fixed set of attributes, so every mesh works with every mesh pipeline. */
export interface Geometry {
  positions: Floats;
  normals: Floats;
  uvs: Floats;
  extras: Floats;
  indices: Uint32Array<ArrayBuffer>;
}

/**
 * Vertex attribute locations, shared by GLSL (`layout(location = n)`) and WGSL (`@location(n)`).
 * Per-instance attributes follow from `instance` onwards.
 */
export const ATTR = { position: 0, normal: 1, uv: 2, extra: 3, instance: 4 } as const;

export type TextureFormat = "rgba8" | "rgba16f" | "depth24";

export interface TextureDesc {
  label: string;
  width: number;
  height: number;
  format: TextureFormat;
  /** More than one makes a multisampled render target. */
  samples?: number;
  mipmaps?: boolean;
  wrap?: "clamp" | "repeat";
  filter?: "linear" | "nearest";
  /** Sampled as a shadow map, with a less-or-equal depth comparison. */
  compare?: boolean;
  /** Use anisotropic filtering when the device supports it. */
  anisotropic?: boolean;
}

export interface Texture {
  readonly desc: TextureDesc;
  readonly width: number;
  readonly height: number;
  destroy(): void;
}

/**
 * How a pipeline reads a texture.
 * - `float`: filtered color (`sampler2D` / `texture_2d<f32>` with a sampler).
 * - `shadow`: depth comparison (`sampler2DShadow` / `texture_depth_2d` with a comparison sampler).
 * - `depth`: raw depth values, read without filtering (`sampler2D` / `texture_depth_2d` or
 *   `texture_depth_multisampled_2d` with `textureLoad`).
 */
export interface TextureBinding {
  /** The GLSL sampler uniform's name. WGSL finds it by binding number instead. */
  name: string;
  kind: "float" | "shadow" | "depth";
  /** WebGPU only: the texture is multisampled (WebGL2 always reads a resolved copy). */
  multisampled?: boolean;
}

/**
 * Both shading languages for one material.
 *
 * GLSL gets a `#version 300 es` header. It reads the `Frame` uniform block at binding 0 and a
 * `Params` block at binding 1, and its sampler uniforms are named by the pipeline's textures.
 *
 * WGSL has entry points `vs` and `fs`. Group 0 holds the frame uniforms at binding 0, then for
 * texture `i` the texture at binding `1 + 2i` and its sampler at `2 + 2i` (no sampler for `depth`).
 * Group 1 binding 0 holds the params.
 */
export interface ShaderSource {
  glsl: { vertex: string; fragment: string };
  wgsl: string;
}

export type Blend = "none" | "alpha" | "premultiplied" | "additive";

export type ColorTarget = TextureFormat | "canvas";

export interface PipelineDesc {
  label: string;
  shader: ShaderSource;
  /** `mesh` reads the standard attributes; `none` generates vertices, as a fullscreen triangle does. */
  vertex: "mesh" | "none";
  /** Per-instance attribute sizes, at locations from `ATTR.instance`. Must match the mesh. */
  instances?: readonly number[];
  /** Reads the frame uniforms. */
  frame?: boolean;
  /** Size in bytes of the params block, if the pipeline has one. */
  params?: number;
  textures?: readonly TextureBinding[];
  /** The color attachment's format. Left out for a depth-only pipeline, which has no fragment stage. */
  color?: ColorTarget;
  /** `write` tests and writes depth; `test` only tests. Left out when the pass has no depth. */
  depth?: "write" | "test";
  samples?: number;
  blend?: Blend;
  alphaToCoverage?: boolean;
}

export interface Pipeline {
  readonly desc: PipelineDesc;
}

export interface Mesh {
  /** Replaces the per-instance data. Only for meshes created with an instance layout. */
  setInstances(data: Floats, count: number): void;
}

/** A slice of this frame's uniform memory. It is only valid until the frame ends. */
export interface Uniforms {
  readonly offset: number;
  readonly size: number;
}

export interface DrawBindings {
  frame?: Uniforms;
  params?: Uniforms;
  textures?: readonly Texture[];
}

export interface PassDesc {
  label: string;
  color?: {
    target: Texture | "canvas";
    /** Mip level to render into. */
    level?: number;
    /** Where a multisampled target resolves to when the pass ends. */
    resolve?: Texture;
    clear?: readonly [number, number, number, number];
  };
  depth?: {
    target: Texture;
    /** Ignored when the device can't resolve depth; check `caps.depthResolve`. */
    resolve?: Texture;
    clear?: number;
  };
}

export interface Pass {
  /** Draws a mesh, or a single generated triangle when `mesh` is null. */
  draw(pipeline: Pipeline, mesh: Mesh | null, bindings?: DrawBindings): void;
  end(): void;
}

export interface Capabilities {
  backend: BackendKind;
  /** What the browser says about the GPU, as far as it says anything. */
  adapter: string;
  /** The best renderable format for the HDR scene: `rgba16f`, or `rgba8` without float targets. */
  hdrFormat: "rgba16f" | "rgba8";
  /** Most MSAA samples the HDR format supports, capped at 4. */
  samples: number;
  /** Whether a multisampled depth buffer can be resolved into a sampled texture. */
  depthResolve: boolean;
  /** Anisotropic filtering level used for `anisotropic` textures, or 1 without it. */
  anisotropy: number;
  /** WebGL2 extensions or WebGPU features this device exposes. */
  features: readonly string[];
  /** The subset of `features` the renderer actually uses. */
  used: readonly string[];
}

export interface Device {
  readonly caps: Capabilities;
  createMesh(geometry: Geometry, instances?: readonly number[]): Mesh;
  createTexture(desc: TextureDesc): Texture;
  createPipeline(desc: PipelineDesc): Pipeline;
  /** Starts recording a frame. Uniforms and passes belong to the current frame. */
  beginFrame(): void;
  /** Copies data into this frame's uniform memory. */
  uniforms(data: Floats): Uniforms;
  beginPass(desc: PassDesc): Pass;
  /** Fills every mip level below the first from the one above it. */
  generateMipmaps(texture: Texture): void;
  /** Submits the frame. */
  endFrame(): void;
  /** Resolves once every pipeline has compiled, or rejects with the first error. */
  validate(): Promise<void>;
  onLost(callback: (message: string) => void): void;
  destroy(): void;
}
