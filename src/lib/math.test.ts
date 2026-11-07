import { describe, expect, it } from "vitest";
import { expectVecClose, IDENTITY } from "../test/matchers";
import {
  add,
  clamp,
  cross,
  dot,
  invert,
  length,
  lerp3,
  lookAt,
  mix,
  multiply,
  normalize,
  ortho,
  perspective,
  rayBox,
  scale,
  smoothstep,
  sub,
  transformPoint,
  type Vec3,
} from "./math";

describe("scalar helpers", () => {
  it("clamps into the range", () => {
    expect(clamp(-1, 0, 1)).toBe(0);
    expect(clamp(0.25, 0, 1)).toBe(0.25);
    expect(clamp(9, 0, 1)).toBe(1);
  });

  it("mixes linearly, including past the ends", () => {
    expect(mix(2, 4, 0)).toBe(2);
    expect(mix(2, 4, 0.5)).toBe(3);
    expect(mix(2, 4, 1.5)).toBe(5);
  });

  it("smoothsteps like GLSL", () => {
    expect(smoothstep(0, 1, -1)).toBe(0);
    expect(smoothstep(0, 1, 0.5)).toBe(0.5);
    expect(smoothstep(0, 1, 2)).toBe(1);
    // A reversed edge pair flips the ramp, which layout.ts relies on.
    expect(smoothstep(1, 0, 0.2)).toBeCloseTo(1 - smoothstep(0, 1, 0.2));
  });
});

describe("vectors", () => {
  const a: Vec3 = [1, 2, 3];
  const b: Vec3 = [-4, 0.5, 2];

  it("adds, subtracts and scales per component", () => {
    expect(add(a, b)).toEqual([-3, 2.5, 5]);
    expect(sub(a, b)).toEqual([5, 1.5, 1]);
    expect(scale(a, 2)).toEqual([2, 4, 6]);
    expect(lerp3(a, b, 0.5)).toEqual([-1.5, 1.25, 2.5]);
  });

  it("follows the right-hand rule for cross products", () => {
    expect(cross([1, 0, 0], [0, 1, 0])).toEqual([0, 0, 1]);
    const c = cross(a, b);
    expect(dot(c, a)).toBeCloseTo(0);
    expect(dot(c, b)).toBeCloseTo(0);
  });

  it("normalizes to unit length", () => {
    expect(length(normalize(b))).toBeCloseTo(1);
    expect(length([3, 4, 0])).toBe(5);
  });

  it("leaves the zero vector alone instead of producing NaN", () => {
    expect(normalize([0, 0, 0])).toEqual([0, 0, 0]);
  });
});

describe("matrices", () => {
  const view = lookAt([1.5, 2, 4], [0, 0.6, 0], [0, 1, 0]);
  const proj = perspective(Math.PI / 4, 16 / 9, 0.1, 40);
  const viewProj = multiply(proj, view);

  it("treats the identity as neutral", () => {
    expectVecClose(multiply(IDENTITY, view), view);
    expectVecClose(multiply(view, IDENTITY), view);
  });

  it("inverts a view-projection", () => {
    expectVecClose(multiply(viewProj, invert(viewProj)), IDENTITY, 4);
    expectVecClose(multiply(invert(view), view), IDENTITY, 5);
  });

  it("puts the eye at the origin, looking down -z", () => {
    expectVecClose(transformPoint(view, [1.5, 2, 4]), [0, 0, 0]);
    const target = transformPoint(view, [0, 0.6, 0]);
    expect(target[0]).toBeCloseTo(0);
    expect(target[1]).toBeCloseTo(0);
    expect(target[2]).toBeCloseTo(-Math.hypot(1.5, 1.4, 4));
  });

  it("maps the near and far planes to GL's -1 and 1 depth", () => {
    expect(transformPoint(proj, [0, 0, -0.1])[2]).toBeCloseTo(-1);
    expect(transformPoint(proj, [0, 0, -40])[2]).toBeCloseTo(1, 4);
  });

  it("maps the edge of the field of view to the edge of clip space", () => {
    const p = perspective(Math.PI / 2, 2, 1, 10);
    // At 90 degrees vertically, a point at depth 5 is on the top edge when y equals 5.
    expect(transformPoint(p, [0, 5, -5])[1]).toBeCloseTo(1);
    expect(transformPoint(p, [10, 0, -5])[0]).toBeCloseTo(1);
  });

  it("maps an orthographic box onto the unit cube", () => {
    const o = ortho(-2, 2, -1, 3, 0.5, 10);
    expectVecClose(transformPoint(o, [-2, -1, -0.5]), [-1, -1, -1]);
    expectVecClose(transformPoint(o, [2, 3, -10]), [1, 1, 1]);
  });
});

describe("rayBox", () => {
  const min: Vec3 = [-1, -1, -1];
  const max: Vec3 = [1, 1, 1];

  it("returns the entry and exit distances of a hit", () => {
    expect(rayBox([-5, 0, 0], [1, 0, 0], min, max)).toEqual({ near: 4, far: 6 });
  });

  it("returns a negative entry distance from inside the box", () => {
    expect(rayBox([0, 0, 0], [0, 1, 0], min, max)).toEqual({ near: -1, far: 1 });
  });

  it("misses a box to the side or behind the ray", () => {
    expect(rayBox([-5, 3, 0], [1, 0, 0], min, max)).toBeNull();
    expect(rayBox([5, 0, 0], [1, 0, 0], min, max)).toBeNull();
  });

  it("handles axis-parallel rays, whose inverse direction is infinite", () => {
    const hit = rayBox([0.5, 0.5, 5], normalize([0, 0, -1]), min, max);
    expect(hit?.near).toBeCloseTo(4);
    expect(hit?.far).toBeCloseTo(6);
  });
});
