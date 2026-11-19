import { describe, expect, it } from "vitest";
import { createRng, fbm3, noise3 } from "./noise";

const take = (rng: () => number, n: number): number[] => Array.from({ length: n }, rng);

describe("createRng", () => {
  // Fixed seeds build the world — the fish scatter, food drops, pebble fields
  // and plant placement all derive from them — so a changed sequence silently
  // regenerates the aquascape. Pin the outputs; change them only on purpose.
  it("produces the documented sequence for a given seed", () => {
    expect(take(createRng(0), 6).map((v) => v.toFixed(12))).toEqual([
      "0.266429208685",
      "0.000329745701",
      "0.223272027448",
      "0.146202147938",
      "0.467327822931",
      "0.545049082721",
    ]);
    expect(take(createRng(90210), 6).map((v) => v.toFixed(12))).toEqual([
      "0.281729944982",
      "0.736178235151",
      "0.921530475374",
      "0.171728212619",
      "0.761466322932",
      "0.374760675011",
    ]);
  });

  it("repeats the same sequence for the same seed", () => {
    expect(take(createRng(42), 20)).toEqual(take(createRng(42), 20));
  });

  it("gives different sequences for different seeds", () => {
    expect(take(createRng(1), 5)).not.toEqual(take(createRng(2), 5));
  });

  it("stays in [0, 1) and is roughly uniform", () => {
    const values = take(createRng(7), 10_000);
    expect(Math.min(...values)).toBeGreaterThanOrEqual(0);
    expect(Math.max(...values)).toBeLessThan(1);
    const mean = values.reduce((sum, v) => sum + v, 0) / values.length;
    expect(mean).toBeCloseTo(0.5, 1);
    const buckets = Array.from({ length: 10 }, () => 0);
    for (const v of values) buckets[Math.floor(v * 10)]++;
    for (const count of buckets) expect(count).toBeGreaterThan(850);
  });
});

describe("noise3", () => {
  const samples = take(createRng(3), 600).map((v, i) => [v * 40 - 20, (i % 37) * 0.7, i * 0.13]);

  it("stays within 0..1", () => {
    for (const [x, y, z] of samples) {
      const n = noise3(x, y, z);
      expect(n).toBeGreaterThanOrEqual(0);
      expect(n).toBeLessThanOrEqual(1);
    }
  });

  it("is deterministic", () => {
    expect(noise3(1.25, -3.5, 8.75)).toBe(noise3(1.25, -3.5, 8.75));
  });

  it("is continuous, including across lattice cells", () => {
    for (const [x, y, z] of [...samples.slice(0, 50), [2, 3, 4]]) {
      expect(Math.abs(noise3(x + 1e-4, y, z) - noise3(x - 1e-4, y, z))).toBeLessThan(1e-3);
    }
  });

  it("actually varies", () => {
    const values = new Set(samples.map(([x, y, z]) => noise3(x, y, z).toFixed(3)));
    expect(values.size).toBeGreaterThan(100);
  });
});

describe("fbm3", () => {
  it("equals plain noise with one octave", () => {
    expect(fbm3(0.3, 1.7, -2.2, 1)).toBe(noise3(0.3, 1.7, -2.2));
  });

  it("stays within 0..1 for several octave counts", () => {
    for (const octaves of [2, 4, 6]) {
      for (let i = 0; i < 200; i++) {
        const n = fbm3(i * 0.37, i * -0.11, i * 0.05, octaves);
        expect(n).toBeGreaterThanOrEqual(0);
        expect(n).toBeLessThanOrEqual(1);
      }
    }
  });
});
