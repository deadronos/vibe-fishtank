import { describe, expect, it } from "vitest";
import {
  clamp,
  cross,
  dot,
  invert,
  lookAt,
  mix,
  multiply,
  normalize,
  ortho,
  perspective,
  rayBox,
  smoothstep,
  sub,
  transformPoint,
  type Floats,
} from "./math";

const near = (actual: number, expected: number, tolerance = 1e-6): void => {
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(tolerance);
};

describe("scalar helpers", () => {
  it("clamps to the range", () => {
    expect(clamp(0.5, 0, 1)).toBe(0.5);
    expect(clamp(-1, 0, 1)).toBe(0);
    expect(clamp(2, 0, 1)).toBe(1);
  });

  it("mixes endpoints exactly at 0 and 1", () => {
    expect(mix(2, 10, 0)).toBe(2);
    expect(mix(2, 10, 1)).toBe(10);
    expect(mix(2, 10, 0.5)).toBe(6);
  });

  it("smoothstep is symmetric, monotone and clamped", () => {
    expect(smoothstep(0, 1, 0.5)).toBeCloseTo(0.5, 12);
    expect(smoothstep(0, 1, 0.75)).toBeCloseTo(1 - smoothstep(0, 1, 0.25), 12);
    expect(smoothstep(0, 1, -1)).toBe(0);
    expect(smoothstep(0, 1, 2)).toBe(1);
    expect(smoothstep(0, 1, 0.4)).toBeLessThan(smoothstep(0, 1, 0.5));
  });
});

describe("vector helpers", () => {
  it("crosses and dots correctly", () => {
    expect(cross([1, 0, 0], [0, 1, 0])).toEqual([0, 0, 1]);
    expect(cross([1, 0, 0], [1, 0, 0])).toEqual([0, 0, 0]);
    expect(dot([1, 2, 3], [4, 5, 6])).toBe(32);
  });

  it("normalizes, and survives the zero vector", () => {
    expect(normalize([3, 0, 0])).toEqual([1, 0, 0]);
    const zero = normalize([0, 0, 0]);
    expect(Number.isFinite(zero[0] + zero[1] + zero[2])).toBe(true);
    const unit = normalize([1, 2, 3]);
    expect(Math.hypot(unit[0], unit[1], unit[2])).toBeCloseTo(1, 12);
  });

  it("subtracts", () => {
    expect(sub([5, 5, 5], [1, 2, 3])).toEqual([4, 3, 2]);
  });
});

describe("projection matrices", () => {
  it("perspective matches the known closed form", () => {
    // fovY 90°, aspect 2, near 1, far 3.
    const m = perspective(Math.PI / 2, 2, 1, 3);
    near(m[0], 0.5); // f / aspect
    near(m[5], 1); // f
    near(m[10], -2); // (far + near) / (near - far)
    near(m[11], -1);
    near(m[14], -3); // 2·far·near / (near - far)
    near(m[15], 0);
  });

  it("ortho maps its own bounds to the unit cube", () => {
    // near and far are distances along -z, as in OpenGL's glOrtho.
    const m = ortho(-1, 1, -1, 1, 0, 2);
    const nearCorner = transformPoint(m, [-1, -1, 0]);
    const farCorner = transformPoint(m, [1, 1, -2]);
    near(nearCorner[0], -1);
    near(nearCorner[1], -1);
    near(nearCorner[2], -1);
    near(farCorner[0], 1);
    near(farCorner[1], 1);
    near(farCorner[2], 1);
  });
});

describe("lookAt", () => {
  it("moves the target in front of the eye and keeps the world upright", () => {
    const eye: [number, number, number] = [0, 0, 5];
    const view = lookAt(eye, [0, 0, 0], [0, 1, 0]);
    const at = transformPoint(view, [0, 0, 0]);
    near(at[0], 0);
    near(at[1], 0);
    near(at[2], -5);
    const above = transformPoint(view, [0, 1, 0]);
    near(above[0], 0);
    near(above[1], 1);
    near(above[2], -5);
    const right = transformPoint(view, [1, 0, 0]);
    near(right[0], 1);
    near(right[1], 0);
    near(right[2], -5);
  });
});

const identity = (): Floats => {
  const m = new Float32Array(16);
  m[0] = m[5] = m[10] = m[15] = 1;
  return m;
};

describe("multiply and invert", () => {
  it("multiplies by the identity without change", () => {
    const m = multiply(perspective(1, 1.7, 0.1, 40), lookAt([0, 1, 4], [0, 0, 0], [0, 1, 0]));
    const product = multiply(identity(), m);
    for (let i = 0; i < 16; i++) near(product[i], m[i], 1e-6);
  });

  it("round-trips a real view-projection through invert", () => {
    const vp = multiply(
      perspective((34 * Math.PI) / 180, 1.5, 0.1, 40),
      lookAt([0.5, 0.8, 3], [0, 0.6, 0], [0, 1, 0]),
    );
    const product = multiply(vp, invert(vp));
    for (let c = 0; c < 4; c++) {
      for (let r = 0; r < 4; r++) {
        near(product[c * 4 + r], c === r ? 1 : 0, 1e-4);
      }
    }
  });
});

describe("transformPoint and rayBox together", () => {
  const eye: [number, number, number] = [0, 0.5, 3];
  const vp = multiply(
    perspective((34 * Math.PI) / 180, 1.5, 0.1, 40),
    lookAt(eye, [0, 0.5, 0], [0, 1, 0]),
  );

  it("reconstructs a world point from its screen position, as the picking ray does", () => {
    const world: [number, number, number] = [0.25, 0.9, -0.4];
    const ndc = transformPoint(vp, world);
    const onFarPlane = transformPoint(invert(vp), [ndc[0], ndc[1], 1]);
    // The reconstructed point lies on the ray from the eye through the original.
    const ray = sub(onFarPlane, eye);
    const toWorld = sub(world, eye);
    const area = cross(ray, toWorld);
    expect(Math.hypot(area[0], area[1], area[2])).toBeLessThan(1e-4);
    expect(dot(ray, toWorld)).toBeGreaterThan(0);
  });

  it("rayBox hits boxes in front of the ray", () => {
    const hit = rayBox([0, 0, 0], [1, 0, 0], [1, -1, -1], [2, 1, 1]);
    expect(hit).not.toBeNull();
    near(hit!.near, 1);
    near(hit!.far, 2);
  });

  it("rayBox enters through the near face of a thin slab", () => {
    // Looking straight down into a water-volume-like flat box.
    const hit = rayBox([0.5, 5, 0], [0, -1, 0], [-1, 0, -1], [1, 0.1, 1]);
    expect(hit).not.toBeNull();
    near(hit!.near, 4.9);
    near(hit!.far, 5);
  });

  it("rayBox misses, and reports both slabs when starting inside", () => {
    expect(rayBox([0, 0, 0], [0, 0, 1], [1, -1, -1], [2, 1, 1])).toBeNull();
    expect(rayBox([0, 0, 0], [-1, 0, 0], [1, -1, -1], [2, 1, 1])).toBeNull(); // box is behind
    const inside = rayBox([0, 0, 0], [1, 0, 0], [-1, -1, -1], [1, 1, 1]);
    expect(inside).not.toBeNull();
    near(inside!.near, -1);
    near(inside!.far, 1);
  });
});
