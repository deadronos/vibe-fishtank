/** Picks a backend: WebGPU when it works, WebGL2 otherwise. */

import type { BackendKind, Capabilities, Device } from "./types";
import { createWebGL2Device } from "./webgl2";
import { createWebGPUDevice } from "./webgpu";

export type * from "./types";
export { ATTR } from "./types";

export type Preference = "auto" | BackendKind;

/** Reads `?renderer=webgpu` or `?renderer=webgl2` from the page address. */
export function preferredBackend(search: string = location.search): Preference {
  const value = new URLSearchParams(search).get("renderer")?.toLowerCase();
  if (value === "webgpu") return "webgpu";
  if (value === "webgl2" || value === "webgl") return "webgl2";
  return "auto";
}

export interface SkippedBackend {
  backend: BackendKind;
  reason: string;
}

export interface Renderer<T> {
  device: Device;
  /** The canvas in use, which is a replacement if an earlier backend failed. */
  canvas: HTMLCanvasElement;
  app: T;
  /** Backends tried before this one, and why each was passed over. */
  skipped: SkippedBackend[];
}

/**
 * Creates a device and runs `setup` with it, preferring WebGPU unless `preference` says otherwise.
 * If a backend is missing, `setup` throws, or its pipelines fail validation, the next one is tried.
 */
export async function createRenderer<T>(
  canvas: HTMLCanvasElement,
  preference: Preference,
  setup: (device: Device, canvas: HTMLCanvasElement) => T,
): Promise<Renderer<T>> {
  const order: BackendKind[] =
    preference === "webgl2" ? ["webgl2", "webgpu"] : ["webgpu", "webgl2"];
  const skipped: SkippedBackend[] = [];
  let current = canvas;
  for (const backend of order) {
    let device: Device | null = null;
    try {
      device =
        backend === "webgpu" ? await createWebGPUDevice(current) : createWebGL2Device(current);
      if (!device) {
        skipped.push({ backend, reason: "not supported by this browser or GPU" });
        continue;
      }
      const app = setup(device, current);
      await device.validate();
      return { device, canvas: current, app, skipped };
    } catch (error) {
      console.warn(`${backend} failed, trying the next renderer.`, error);
      skipped.push({ backend, reason: error instanceof Error ? error.message : String(error) });
      device?.destroy();
      // A canvas keeps the first kind of context it hands out, so the next try needs a new one.
      const fresh = current.cloneNode(false) as HTMLCanvasElement;
      current.replaceWith(fresh);
      current = fresh;
    }
  }
  const reasons = skipped.map((s) => `${s.backend}: ${s.reason}`).join("\n");
  throw new Error(`No renderer could start.\n${reasons}`);
}

const NAMES: Record<BackendKind, string> = { webgpu: "WebGPU", webgl2: "WebGL2" };

export function backendName(backend: BackendKind): string {
  return NAMES[backend];
}

/** A few lines describing what the device offers and which of it the renderer uses. */
export function describe(caps: Capabilities, skipped: readonly SkippedBackend[] = []): string[] {
  const lines = [
    `${NAMES[caps.backend]} on ${caps.adapter}`,
    `HDR ${caps.hdrFormat} · ${caps.samples}x MSAA · ${caps.anisotropy}x anisotropic filtering`,
  ];
  if (caps.used.length > 0) lines.push(`Using ${caps.used.join(", ")}`);
  for (const s of skipped) lines.push(`Skipped ${NAMES[s.backend]}: ${s.reason.split("\n")[0]}`);
  return lines;
}
