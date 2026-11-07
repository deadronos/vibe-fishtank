import { beforeEach, describe, expect, it, vi } from "vitest";
import type { Capabilities, Device } from "./types";
import { backendName, createRenderer, describe as describeCaps, preferredBackend } from "./index";
import { createWebGL2Device } from "./webgl2";
import { createWebGPUDevice } from "./webgpu";

// The real backends need a GPU; here they're stand-ins whose behaviour each test picks.
vi.mock("./webgpu", () => ({ createWebGPUDevice: vi.fn() }));
vi.mock("./webgl2", () => ({ createWebGL2Device: vi.fn() }));

const webgpu = vi.mocked(createWebGPUDevice);
const webgl2 = vi.mocked(createWebGL2Device);

interface FakeCanvas {
  id: number;
  cloneNode: () => FakeCanvas;
  replaceWith: (other: FakeCanvas) => void;
}

let canvases = 0;
function fakeCanvas(): FakeCanvas {
  const canvas: FakeCanvas = {
    id: canvases++,
    cloneNode: vi.fn(() => fakeCanvas()),
    replaceWith: vi.fn(),
  };
  return canvas;
}
const asCanvas = (c: FakeCanvas) => c as unknown as HTMLCanvasElement;

function fakeDevice(name: string, validate: () => Promise<void> = async () => {}) {
  return { name, validate: vi.fn(validate), destroy: vi.fn() } as unknown as Device & {
    name: string;
    destroy: ReturnType<typeof vi.fn>;
  };
}

describe("preferredBackend", () => {
  it.each([
    ["?renderer=webgpu", "webgpu"],
    ["?renderer=WebGPU", "webgpu"],
    ["?renderer=webgl2", "webgl2"],
    ["?renderer=webgl", "webgl2"],
    ["?renderer=vulkan", "auto"],
    ["?other=1", "auto"],
    ["", "auto"],
  ])("reads %j as %s", (search, expected) => {
    expect(preferredBackend(search)).toBe(expected);
  });
});

describe("createRenderer", () => {
  beforeEach(() => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
  });

  it("uses WebGPU when it works", async () => {
    const device = fakeDevice("gpu");
    webgpu.mockResolvedValue(device as never);
    const canvas = fakeCanvas();
    const setup = vi.fn(() => "app");

    const result = await createRenderer(asCanvas(canvas), "auto", setup);

    expect(result).toEqual({ device, canvas, app: "app", skipped: [] });
    expect(setup).toHaveBeenCalledExactlyOnceWith(device, canvas);
    expect(webgl2).not.toHaveBeenCalled();
  });

  it("falls back on the same canvas when WebGPU isn't available", async () => {
    const gl = fakeDevice("gl");
    webgpu.mockResolvedValue(null);
    webgl2.mockReturnValue(gl as never);
    const canvas = fakeCanvas();

    const result = await createRenderer(asCanvas(canvas), "auto", () => 1);

    expect(result.device).toBe(gl);
    expect(result.canvas).toBe(canvas);
    expect(result.skipped).toEqual([
      { backend: "webgpu", reason: "not supported by this browser or GPU" },
    ]);
  });

  it("swaps in a fresh canvas when setup throws", async () => {
    const gpu = fakeDevice("gpu");
    const gl = fakeDevice("gl");
    webgpu.mockResolvedValue(gpu as never);
    webgl2.mockReturnValue(gl as never);
    const canvas = fakeCanvas();
    const setup = vi.fn((device: Device) => {
      if (device === gpu) throw new Error("bad pipeline");
      return "ok";
    });

    const result = await createRenderer(asCanvas(canvas), "auto", setup);

    expect(gpu.destroy).toHaveBeenCalledOnce();
    expect(result.device).toBe(gl);
    expect(result.canvas).not.toBe(canvas);
    expect(canvas.replaceWith).toHaveBeenCalledExactlyOnceWith(result.canvas);
    expect(webgl2).toHaveBeenCalledWith(result.canvas);
    expect(result.skipped).toEqual([{ backend: "webgpu", reason: "bad pipeline" }]);
    expect(console.warn).toHaveBeenCalledOnce();
  });

  it("falls back when validation finds shader errors", async () => {
    const gpu = fakeDevice("gpu", async () => {
      throw new Error("glass:69:7 unresolved value");
    });
    webgpu.mockResolvedValue(gpu as never);
    webgl2.mockReturnValue(fakeDevice("gl") as never);

    const result = await createRenderer(asCanvas(fakeCanvas()), "auto", () => 0);

    expect(gpu.destroy).toHaveBeenCalledOnce();
    expect(result.skipped[0]).toEqual({ backend: "webgpu", reason: "glass:69:7 unresolved value" });
  });

  it("tries WebGL2 first when asked to", async () => {
    const gl = fakeDevice("gl");
    webgl2.mockReturnValue(gl as never);

    const result = await createRenderer(asCanvas(fakeCanvas()), "webgl2", () => 0);

    expect(result.device).toBe(gl);
    expect(webgpu).not.toHaveBeenCalled();
  });

  it("still tries WebGPU when the requested WebGL2 can't start", async () => {
    const gpu = fakeDevice("gpu");
    webgl2.mockReturnValue(null);
    webgpu.mockResolvedValue(gpu as never);

    const result = await createRenderer(asCanvas(fakeCanvas()), "webgl2", () => 0);

    expect(result.device).toBe(gpu);
    expect(result.skipped.map((s) => s.backend)).toEqual(["webgl2"]);
  });

  it("reports every reason when nothing starts", async () => {
    webgpu.mockRejectedValue(new Error("adapter lost"));
    webgl2.mockImplementation(() => {
      throw "context refused"; // Non-Error throws still produce a readable reason.
    });

    await expect(createRenderer(asCanvas(fakeCanvas()), "auto", () => 0)).rejects.toThrow(
      "No renderer could start.\nwebgpu: adapter lost\nwebgl2: context refused",
    );
  });
});

describe("describe", () => {
  const caps: Capabilities = {
    backend: "webgl2",
    adapter: "Apple M4",
    hdrFormat: "rgba16f",
    samples: 4,
    depthResolve: true,
    anisotropy: 16,
    features: ["EXT_color_buffer_float"],
    used: ["EXT_color_buffer_float", "EXT_texture_filter_anisotropic"],
  };

  it("names the backend", () => {
    expect(backendName("webgpu")).toBe("WebGPU");
    expect(backendName("webgl2")).toBe("WebGL2");
  });

  it("summarizes the device and what it uses", () => {
    expect(describeCaps(caps)).toEqual([
      "WebGL2 on Apple M4",
      "HDR rgba16f · 4x MSAA · 16x anisotropic filtering",
      "Using EXT_color_buffer_float, EXT_texture_filter_anisotropic",
    ]);
  });

  it("leaves out an empty list of extensions and keeps only the first line of a reason", () => {
    const lines = describeCaps({ ...caps, used: [] }, [
      { backend: "webgpu", reason: "shader failed\nat line 3" },
    ]);
    expect(lines).toHaveLength(3);
    expect(lines[2]).toBe("Skipped WebGPU: shader failed");
  });
});
