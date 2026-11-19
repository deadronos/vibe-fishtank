/** The WebGPU backend. A frame is recorded into one command encoder and submitted at the end. */

import type { Floats } from "../math";
import {
  ATTR,
  type Blend,
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
  type TextureFormat,
  type Uniforms,
} from "./types";

// The DOM typings describe WebGPU's interfaces but not its global flag namespaces, so these
// mirror the values fixed by the WebGPU specification.
const BufferUsage = { COPY_DST: 0x8, INDEX: 0x10, VERTEX: 0x20, UNIFORM: 0x40 } as const;
const TextureUsage = { TEXTURE_BINDING: 0x4, RENDER_ATTACHMENT: 0x10 } as const;
const ShaderStage = { VERTEX: 0x1, FRAGMENT: 0x2 } as const;

/** Each uniform allocation gets one slot, which is also WebGPU's dynamic-offset alignment. */
const SLOT_BYTES = 256;
const ARENA_BYTES = 256 * 1024;
const DEPTH_FORMAT: GPUTextureFormat = "depth24plus";

const FORMATS: Record<TextureFormat, GPUTextureFormat> = {
  rgba8: "rgba8unorm",
  rgba16f: "rgba16float",
  depth24: DEPTH_FORMAT,
};

const BLENDS: Record<Exclude<Blend, "none">, GPUBlendComponent> = {
  alpha: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
  premultiplied: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
  additive: { srcFactor: "one", dstFactor: "one", operation: "add" },
};

/** Copies each mip level from the one above it. */
const MIP_WGSL = `
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
struct VOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f }
@vertex fn vs(@builtin(vertex_index) i: u32) -> VOut {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  var o: VOut;
  o.pos = vec4f(p * 2.0 - 1.0, 0.0, 1.0);
  o.uv = vec2f(p.x, 1.0 - p.y);
  return o;
}
@fragment fn fs(v: VOut) -> @location(0) vec4f {
  return textureSample(src, samp, v.uv);
}`;

let nextId = 1;

class WGPUTexture implements Texture {
  readonly id = nextId++;
  readonly desc: TextureDesc;
  readonly texture: GPUTexture;
  readonly view: GPUTextureView;
  readonly sampler: GPUSampler;
  readonly levels: number;
  private readonly levelViews: GPUTextureView[] = [];
  private readonly device: WebGPUDevice;

  constructor(device: WebGPUDevice, desc: TextureDesc) {
    this.device = device;
    this.desc = desc;
    this.levels = desc.mipmaps ? Math.floor(Math.log2(Math.max(desc.width, desc.height))) + 1 : 1;
    this.texture = device.gpu.createTexture({
      label: desc.label,
      size: [desc.width, desc.height],
      format: FORMATS[desc.format],
      sampleCount: desc.samples ?? 1,
      mipLevelCount: this.levels,
      usage: TextureUsage.RENDER_ATTACHMENT | TextureUsage.TEXTURE_BINDING,
    });
    this.view = this.texture.createView();
    const linear = desc.filter !== "nearest";
    const filter: GPUFilterMode = linear ? "linear" : "nearest";
    const address: GPUAddressMode = desc.wrap === "repeat" ? "repeat" : "clamp-to-edge";
    const anisotropic = desc.anisotropic && linear && desc.mipmaps;
    this.sampler = device.gpu.createSampler({
      label: desc.label,
      addressModeU: address,
      addressModeV: address,
      magFilter: filter,
      minFilter: filter,
      mipmapFilter: desc.mipmaps ? "linear" : "nearest",
      compare: desc.compare ? "less-equal" : undefined,
      maxAnisotropy: anisotropic ? device.caps.anisotropy : 1,
    });
  }

  get width(): number {
    return this.desc.width;
  }

  get height(): number {
    return this.desc.height;
  }

  /** A view of a single mip level, for rendering into it. */
  level(level: number): GPUTextureView {
    if (this.levels === 1) return this.view;
    this.levelViews[level] ??= this.texture.createView({ baseMipLevel: level, mipLevelCount: 1 });
    return this.levelViews[level];
  }

  destroy(): void {
    this.device.forget();
    this.texture.destroy();
  }
}

class WGPUPipeline implements Pipeline {
  readonly desc: PipelineDesc;
  readonly pipeline: GPURenderPipeline;
  private readonly device: WebGPUDevice;
  private readonly layout0: GPUBindGroupLayout;
  private readonly paramsGroup: GPUBindGroup | null = null;
  private groups = new Map<string, GPUBindGroup>();
  private generation = -1;

  constructor(device: WebGPUDevice, desc: PipelineDesc) {
    this.device = device;
    this.desc = desc;
    const gpu = device.gpu;
    const both = ShaderStage.VERTEX | ShaderStage.FRAGMENT;
    const dynamicUniform: GPUBufferBindingLayout = { type: "uniform", hasDynamicOffset: true };

    const entries: GPUBindGroupLayoutEntry[] = [];
    if (desc.frame) entries.push({ binding: 0, visibility: both, buffer: dynamicUniform });
    desc.textures?.forEach((texture, i) => {
      const binding = 1 + i * 2;
      const visibility = ShaderStage.FRAGMENT;
      if (texture.kind === "float") {
        entries.push({ binding, visibility, texture: { sampleType: "float" } });
        entries.push({ binding: binding + 1, visibility, sampler: { type: "filtering" } });
      } else if (texture.kind === "shadow") {
        entries.push({ binding, visibility, texture: { sampleType: "depth" } });
        entries.push({ binding: binding + 1, visibility, sampler: { type: "comparison" } });
      } else {
        entries.push({
          binding,
          visibility,
          texture: { sampleType: "depth", multisampled: texture.multisampled ?? false },
        });
      }
    });
    this.layout0 = gpu.createBindGroupLayout({ label: `${desc.label} group 0`, entries });
    const layouts = [this.layout0];
    if (desc.params) {
      const layout1 = gpu.createBindGroupLayout({
        label: `${desc.label} params`,
        entries: [{ binding: 0, visibility: both, buffer: dynamicUniform }],
      });
      layouts.push(layout1);
      this.paramsGroup = gpu.createBindGroup({
        layout: layout1,
        entries: [{ binding: 0, resource: { buffer: device.arena, size: SLOT_BYTES } }],
      });
    }

    const module = device.module(desc.shader.wgsl, desc.label);
    const buffers: GPUVertexBufferLayout[] = [];
    if (desc.vertex === "mesh") {
      const single = (location: number, components: number): GPUVertexBufferLayout => ({
        arrayStride: components * 4,
        attributes: [{ shaderLocation: location, offset: 0, format: floatFormat(components) }],
      });
      buffers.push(
        single(ATTR.position, 3),
        single(ATTR.normal, 3),
        single(ATTR.uv, 2),
        single(ATTR.extra, 1),
      );
      if (desc.instances) {
        let offset = 0;
        const attributes = desc.instances.map((size, i): GPUVertexAttribute => {
          const attribute = {
            shaderLocation: ATTR.instance + i,
            offset,
            format: floatFormat(size),
          };
          offset += size * 4;
          return attribute;
        });
        buffers.push({ arrayStride: offset, stepMode: "instance", attributes });
      }
    }

    const blend = desc.blend && desc.blend !== "none" ? BLENDS[desc.blend] : undefined;
    this.pipeline = gpu.createRenderPipeline({
      label: desc.label,
      layout: gpu.createPipelineLayout({ bindGroupLayouts: layouts }),
      vertex: { module, entryPoint: "vs", buffers },
      fragment: desc.color
        ? {
            module,
            entryPoint: "fs",
            targets: [
              {
                format: desc.color === "canvas" ? device.canvasFormat : FORMATS[desc.color],
                blend: blend ? { color: blend, alpha: blend } : undefined,
              },
            ],
          }
        : undefined,
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: desc.depth
        ? {
            format: DEPTH_FORMAT,
            depthWriteEnabled: desc.depth === "write",
            depthCompare: "less",
          }
        : undefined,
      multisample: {
        count: desc.samples ?? 1,
        alphaToCoverageEnabled: desc.alphaToCoverage ?? false,
      },
    });
  }

  bind(pass: GPURenderPassEncoder, bindings: DrawBindings): void {
    const { desc } = this;
    if (this.generation !== this.device.generation) {
      this.groups.clear();
      this.generation = this.device.generation;
    }
    const textures = (bindings.textures ?? []) as readonly WGPUTexture[];
    const key = textures.map((texture) => texture.id).join(",");
    let group = this.groups.get(key);
    if (!group) {
      const entries: GPUBindGroupEntry[] = [];
      if (desc.frame) {
        entries.push({ binding: 0, resource: { buffer: this.device.arena, size: SLOT_BYTES } });
      }
      textures.forEach((texture, i) => {
        entries.push({ binding: 1 + i * 2, resource: texture.view });
        if (desc.textures?.[i].kind !== "depth") {
          entries.push({ binding: 2 + i * 2, resource: texture.sampler });
        }
      });
      group = this.device.gpu.createBindGroup({ layout: this.layout0, entries });
      this.groups.set(key, group);
    }
    if (desc.frame) {
      if (!bindings.frame) throw new Error(`${desc.label} needs frame uniforms`);
      pass.setBindGroup(0, group, [bindings.frame.offset]);
    } else {
      pass.setBindGroup(0, group);
    }
    if (this.paramsGroup) {
      if (!bindings.params) throw new Error(`${desc.label} needs params`);
      pass.setBindGroup(1, this.paramsGroup, [bindings.params.offset]);
    }
  }
}

function floatFormat(components: number): GPUVertexFormat {
  return components === 1 ? "float32" : (`float32x${components}` as GPUVertexFormat);
}

class WGPUMesh implements Mesh {
  readonly buffers: GPUBuffer[];
  readonly indices: GPUBuffer;
  readonly count: number;
  instanceCount = -1;
  instanceBuffer: GPUBuffer | null = null;
  private readonly device: GPUDevice;

  constructor(device: GPUDevice, geometry: Geometry, instances?: readonly number[]) {
    this.device = device;
    const upload = (data: Floats | Uint32Array<ArrayBuffer>, usage: number): GPUBuffer => {
      const buffer = device.createBuffer({
        size: Math.max(4, Math.ceil(data.byteLength / 4) * 4),
        usage: usage | BufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(buffer, 0, data);
      return buffer;
    };
    const vertex = BufferUsage.VERTEX;
    this.buffers = [
      upload(geometry.positions, vertex),
      upload(geometry.normals, vertex),
      upload(geometry.uvs, vertex),
      upload(geometry.extras, vertex),
    ];
    this.indices = upload(geometry.indices, BufferUsage.INDEX);
    this.count = geometry.indices.length;
    if (instances) this.instanceCount = 0;
  }

  setInstances(data: Floats, count: number): void {
    this.instanceCount = count;
    if (count === 0) return;
    if (!this.instanceBuffer || this.instanceBuffer.size < data.byteLength) {
      this.instanceBuffer?.destroy();
      this.instanceBuffer = this.device.createBuffer({
        size: data.byteLength,
        usage: BufferUsage.VERTEX | BufferUsage.COPY_DST,
      });
    }
    this.device.queue.writeBuffer(this.instanceBuffer, 0, data);
  }
}

class WGPUPass implements Pass {
  private readonly encoder: GPURenderPassEncoder;

  constructor(device: WebGPUDevice, desc: PassDesc) {
    const colorAttachments: GPURenderPassColorAttachment[] = [];
    if (desc.color) {
      const { target, level, resolve, clear } = desc.color;
      colorAttachments.push({
        view: target === "canvas" ? device.canvasView() : (target as WGPUTexture).level(level ?? 0),
        resolveTarget: resolve ? (resolve as WGPUTexture).view : undefined,
        loadOp: clear ? "clear" : "load",
        clearValue: clear ? { r: clear[0], g: clear[1], b: clear[2], a: clear[3] } : undefined,
        // A resolved MSAA buffer is never read again, so tiled GPUs needn't write it out.
        storeOp: resolve ? "discard" : "store",
      });
    }
    const depth = desc.depth;
    this.encoder = device.encoder().beginRenderPass({
      label: desc.label,
      colorAttachments,
      depthStencilAttachment: depth
        ? {
            view: (depth.target as WGPUTexture).view,
            depthLoadOp: depth.clear === undefined ? "load" : "clear",
            depthClearValue: depth.clear,
            depthStoreOp: "store",
          }
        : undefined,
    });
  }

  draw(pipeline: Pipeline, mesh: Mesh | null, bindings: DrawBindings = {}): void {
    const pass = this.encoder;
    const m = mesh as WGPUMesh | null;
    if (m && m.instanceCount === 0) return;
    const p = pipeline as WGPUPipeline;
    pass.setPipeline(p.pipeline);
    p.bind(pass, bindings);
    if (!m) {
      pass.draw(3);
      return;
    }
    m.buffers.forEach((buffer, slot) => pass.setVertexBuffer(slot, buffer));
    if (m.instanceBuffer) pass.setVertexBuffer(ATTR.instance, m.instanceBuffer);
    pass.setIndexBuffer(m.indices, "uint32");
    pass.drawIndexed(m.count, Math.max(1, m.instanceCount));
  }

  end(): void {
    this.encoder.end();
  }
}

export class WebGPUDevice implements Device {
  readonly gpu: GPUDevice;
  readonly caps: Capabilities;
  readonly arena: GPUBuffer;
  readonly canvasFormat: GPUTextureFormat;
  /** Bumped whenever a texture is destroyed, so pipelines drop bind groups that use it. */
  generation = 0;
  private readonly context: GPUCanvasContext;
  private readonly arenaData = new ArrayBuffer(ARENA_BYTES);
  private readonly modules = new Map<string, { module: GPUShaderModule; label: string }>();
  private readonly mipPipelines = new Map<GPUTextureFormat, GPURenderPipeline>();
  private readonly mipSampler: GPUSampler;
  private arenaOffset = 0;
  private frameEncoder: GPUCommandEncoder | null = null;
  private currentView: GPUTextureView | null = null;
  private destroyed = false;
  private reportedErrors = 0;

  constructor(adapter: GPUAdapter, gpu: GPUDevice, context: GPUCanvasContext) {
    this.gpu = gpu;
    this.context = context;
    // Every pipeline is created under this scope, which `validate` closes.
    gpu.pushErrorScope("validation");
    this.canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device: gpu, format: this.canvasFormat, alphaMode: "opaque" });
    this.arena = gpu.createBuffer({
      label: "uniforms",
      size: ARENA_BYTES,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    this.mipSampler = gpu.createSampler({ minFilter: "linear", magFilter: "linear" });
    const info = adapter.info;
    const adapterName =
      info.description || [info.vendor, info.architecture].filter(Boolean).join(" ");
    this.caps = {
      backend: "webgpu",
      adapter: adapterName || "unknown",
      hdrFormat: "rgba16f",
      samples: 4,
      depthResolve: false,
      anisotropy: 8,
      features: [...adapter.features].toSorted(),
      used: [],
    };
  }

  createMesh(geometry: Geometry, instances?: readonly number[]): Mesh {
    return new WGPUMesh(this.gpu, geometry, instances);
  }

  createTexture(desc: TextureDesc): Texture {
    return new WGPUTexture(this, desc);
  }

  createPipeline(desc: PipelineDesc): Pipeline {
    return new WGPUPipeline(this, desc);
  }

  module(code: string, label: string): GPUShaderModule {
    let entry = this.modules.get(code);
    if (!entry) {
      entry = { module: this.gpu.createShaderModule({ label, code }), label };
      this.modules.set(code, entry);
    }
    return entry.module;
  }

  beginFrame(): void {
    this.arenaOffset = 0;
    this.currentView = null;
    this.frameEncoder = this.gpu.createCommandEncoder();
  }

  encoder(): GPUCommandEncoder {
    if (!this.frameEncoder) throw new Error("beginFrame wasn't called");
    return this.frameEncoder;
  }

  canvasView(): GPUTextureView {
    this.currentView ??= this.context.getCurrentTexture().createView();
    return this.currentView;
  }

  uniforms(data: Floats): Uniforms {
    const size = data.byteLength;
    if (size > SLOT_BYTES) throw new Error(`Uniform data is larger than ${SLOT_BYTES} bytes`);
    const offset = this.arenaOffset;
    if (offset + SLOT_BYTES > ARENA_BYTES) throw new Error("Out of uniform memory this frame");
    new Float32Array(this.arenaData, offset, data.length).set(data);
    this.arenaOffset = offset + SLOT_BYTES;
    return { offset, size };
  }

  beginPass(desc: PassDesc): Pass {
    return new WGPUPass(this, desc);
  }

  generateMipmaps(texture: Texture): void {
    const source = texture as WGPUTexture;
    const format = FORMATS[source.desc.format];
    let pipeline = this.mipPipelines.get(format);
    if (!pipeline) {
      const module = this.module(MIP_WGSL, "mipmaps");
      pipeline = this.gpu.createRenderPipeline({
        label: `mipmaps ${format}`,
        layout: "auto",
        vertex: { module, entryPoint: "vs" },
        fragment: { module, entryPoint: "fs", targets: [{ format }] },
      });
      this.mipPipelines.set(format, pipeline);
    }
    const encoder = this.encoder();
    for (let level = 1; level < source.levels; level++) {
      const group = this.gpu.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: source.level(level - 1) },
          { binding: 1, resource: this.mipSampler },
        ],
      });
      const pass = encoder.beginRenderPass({
        label: `mip ${level}`,
        colorAttachments: [{ view: source.level(level), loadOp: "clear", storeOp: "store" }],
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, group);
      pass.draw(3);
      pass.end();
    }
  }

  endFrame(): void {
    const { gpu } = this;
    if (this.arenaOffset > 0)
      gpu.queue.writeBuffer(this.arena, 0, this.arenaData, 0, this.arenaOffset);
    gpu.queue.submit([this.encoder().finish()]);
    this.frameEncoder = null;
  }

  async validate(): Promise<void> {
    const error = await this.gpu.popErrorScope();
    // Shader errors come with line numbers here, rather than in the pipeline error.
    const messages: string[] = [];
    for (const { module, label } of this.modules.values()) {
      const info = await module.getCompilationInfo();
      for (const message of info.messages) {
        if (message.type !== "error") continue;
        messages.push(`${label}:${message.lineNum}:${message.linePos} ${message.message}`);
      }
    }
    this.gpu.addEventListener("uncapturederror", (event) => {
      if (this.reportedErrors++ < 5)
        console.error((event as GPUUncapturedErrorEvent).error.message);
    });
    if (error || messages.length > 0) {
      throw new Error([...messages, error?.message ?? ""].filter(Boolean).join("\n"));
    }
  }

  onLost(callback: (message: string) => void): void {
    void this.gpu.lost.then((info) => {
      if (!this.destroyed) callback(info.message || "The WebGPU device was lost.");
    });
  }

  destroy(): void {
    this.destroyed = true;
    this.context.unconfigure();
    this.gpu.destroy();
  }

  forget(): void {
    this.generation++;
  }
}

/** Resolves to null when WebGPU isn't available, so the caller can fall back to WebGL2. */
export async function createWebGPUDevice(canvas: HTMLCanvasElement): Promise<WebGPUDevice | null> {
  if (!("gpu" in navigator) || !navigator.gpu) return null;
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) return null;
  const device = await adapter.requestDevice({ label: "vibe-fishtank" });
  const context = canvas.getContext("webgpu") as GPUCanvasContext | null;
  if (!context) {
    device.destroy();
    return null;
  }
  return new WebGPUDevice(adapter, device, context);
}
