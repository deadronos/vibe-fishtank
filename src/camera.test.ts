import { describe, expect, it } from "vitest";
import { OrbitCamera } from "./camera";
import { dot, length, multiply, normalize, sub, transformPoint, type Vec3 } from "./lib/math";
import { expectVecClose, IDENTITY } from "./test/matchers";

const TARGET: Vec3 = [0, 0.62, 0];

function settled(camera: OrbitCamera, aspect = 16 / 9, seconds = 3): OrbitCamera {
  for (let t = 0; t < seconds; t += 1 / 60) camera.update(1 / 60, aspect);
  return camera;
}

const distance = (camera: OrbitCamera) => length(sub(camera.position, TARGET));

/** How far the camera moves over 20 idle seconds, once it has settled. */
function idleDrift(reducedMotion: boolean): number {
  const camera = settled(new OrbitCamera(reducedMotion));
  const before = camera.position;
  settled(camera, 16 / 9, 20);
  return length(sub(camera.position, before));
}

describe("OrbitCamera", () => {
  it("looks at the tank, which lands in the middle of the screen", () => {
    const camera = settled(new OrbitCamera(true));
    const [x, y, z] = transformPoint(camera.viewProj, TARGET);
    expect(x).toBeCloseTo(0);
    expect(y).toBeCloseTo(0);
    expect(z).toBeGreaterThan(-1);
    expect(z).toBeLessThan(1);
  });

  it("keeps its inverse view-projection in sync", () => {
    const camera = settled(new OrbitCamera(true));
    expectVecClose(multiply(camera.viewProj, camera.invViewProj), IDENTITY, 3);
  });

  it("keeps an orthonormal basis", () => {
    const camera = settled(new OrbitCamera(true));
    expect(length(camera.right)).toBeCloseTo(1);
    expect(length(camera.up)).toBeCloseTo(1);
    expect(dot(camera.right, camera.up)).toBeCloseTo(0);
    expect(camera.up[1]).toBeGreaterThan(0);
  });

  it("casts a ray through the screen centre towards the tank", () => {
    const camera = settled(new OrbitCamera(true));
    const towards = normalize(sub(TARGET, camera.position));
    expect(dot(camera.ray(0, 0), towards)).toBeCloseTo(1, 4);
  });

  it("zooms in and out within limits", () => {
    const camera = settled(new OrbitCamera(true));
    const start = distance(camera);
    camera.dolly(0.8);
    settled(camera);
    expect(distance(camera)).toBeLessThan(start);
    for (let i = 0; i < 30; i++) camera.dolly(0.5);
    settled(camera);
    expect(distance(camera)).toBeGreaterThanOrEqual(2.3 - 1e-6);
  });

  it("clamps orbiting so the camera never goes under the sand or behind the tank", () => {
    const camera = settled(new OrbitCamera(true));
    camera.orbit(-1e5, -1e5);
    settled(camera);
    expect(camera.position[1]).toBeGreaterThan(0);
    expect(camera.position[2]).toBeGreaterThan(0);
  });

  it("backs off on narrow screens so the whole tank fits", () => {
    const wide = distance(settled(new OrbitCamera(true), 16 / 9));
    const narrow = distance(settled(new OrbitCamera(true), 9 / 19.5));
    expect(narrow).toBeGreaterThan(wide);
  });

  it("drifts when left alone, unless motion is reduced", () => {
    expect(idleDrift(true)).toBeCloseTo(0, 6);
    expect(idleDrift(false)).toBeGreaterThan(0.01);
  });
});
