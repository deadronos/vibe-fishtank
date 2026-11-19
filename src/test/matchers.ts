/** Shared assertions for floating-point vectors and matrices. */

import { expect } from "vitest";
import type { Vec3 } from "../lib/math";

export function expectVecClose(actual: ArrayLike<number>, expected: ArrayLike<number>, digits = 5) {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) {
    expect(actual[i], `component ${i}`).toBeCloseTo(expected[i], digits);
  }
}

export const IDENTITY = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);

export const isFiniteVec = (v: Vec3): boolean => v.every(Number.isFinite);
