import { describe, expect, it } from "vitest";
import { createRng, fbm3, noise3 } from "./noise";

const firstSix = (seed: number): number[] => {
  const rng = createRng(seed);
  return Array.from({ length: 6 }, () => rng());
};

describe("createRng", () => {
  // Fixed seeds build the world: the fish scatter, the pebble field, the plant
  // layout. Changing the sequence silently regenerates all of that content,
  // so pin the outputs the aquascape depends on.
  it("produces the exact documented sequence for a given seed", () => {
    const first = firstSix;
    expect(first(0).map((v) => v.toFixed(12))).toEqual([
      "0.266429208685",
      "0.000329745701",
      "0.223272027448",
      "0.146202147938",
      "0.467327822931",
      "0.545049082721",
    ]);
    expect(first(90210).map((v) => v.toFixed(12))).toEqual([
      "0.281729944982",
      "0.736178235151",
      "0.921530475374",
      "0.171728212619",
      "0.761466322932",
      "0.374760675011",
    ]);
  });

  it("reproduces the same sequence for the same seed", () => {
    const a = createRng(1234);
    const b = createRng(1234);
    for (let i = 0; i < 1000; i++) expect(a()).toBe(b());
  });

  it("produces different sequences for different seeds", () => {
    const a = createRng(1);
    const b = createRng(2);
    let same = 0;
    for (let i = 0; i < 100; i++) if (a() === b()) same++;
    expect(same).toBeLessThan(100);
  });

  it("stays within [0, 1) and is roughly uniform", () => {
    for (const seed of [0, 1, 42, 90210, 0xffffffff]) {
      const rng = createRng(seed);
      let sum = 0;
      const buckets = Array.from({ length: 10 }, () => 0);
      for (let i = 0; i < 10000; i++) {
        const v = rng();
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThan(1);
        sum += v;
        buckets[Math.min(9, Math.floor(v * 10))]++;
      }
      expect(sum / 10000).toBeGreaterThan(0.47);
      expect(sum / 10000).toBeLessThan(0.53);
      // Every decile is populated; no big gaps in the distribution.
      for (const count of buckets) expect(count).toBeGreaterThan(700);
    }
  });
});

describe("noise3", () => {
  it("stays within 0..1", () => {
    const rng = createRng(77);
    for (let i = 0; i < 2000; i++) {
      const v = noise3(rng() * 20 - 10, rng() * 20 - 10, rng() * 20 - 10);
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    }
  });

  it("is continuous: nearby inputs give nearby values", () => {
    const rng = createRng(78);
    for (let i = 0; i < 200; i++) {
      const x = rng() * 8 - 4;
      const y = rng() * 8 - 4;
      const z = rng() * 8 - 4;
      const v = noise3(x, y, z);
      expect(Math.abs(v - noise3(x + 0.01, y, z))).toBeLessThan(0.1);
      expect(Math.abs(v - noise3(x, y + 0.01, z))).toBeLessThan(0.1);
      expect(Math.abs(v - noise3(x, y, z + 0.01))).toBeLessThan(0.1);
    }
  });

  it("is deterministic", () => {
    expect(noise3(1.7, 2.3, 0.9)).toBe(noise3(1.7, 2.3, 0.9));
  });
});

describe("fbm3", () => {
  it("stays within 0..1 and varies over the tank's scale", () => {
    const rng = createRng(79);
    let low = 1;
    let high = 0;
    for (let i = 0; i < 2000; i++) {
      const v = fbm3(rng() * 3 - 1.5, 3, rng() * 2 - 1);
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
      low = Math.min(low, v);
      high = Math.max(high, v);
    }
    expect(high - low).toBeGreaterThan(0.3);
  });

  it("is continuous across small steps in every axis", () => {
    const rng = createRng(80);
    // Four octaves at most 8.4× frequency: a 0.01 step stays well under
    // the sum of each octave's maximum slope, times its weight.
    for (let i = 0; i < 500; i++) {
      const x = rng() * 8 - 4;
      const y = rng() * 8 - 4;
      const z = rng() * 8 - 4;
      const v = fbm3(x, y, z);
      expect(Math.abs(v - fbm3(x + 0.01, y, z))).toBeLessThan(0.06);
      expect(Math.abs(v - fbm3(x, y + 0.01, z))).toBeLessThan(0.06);
      expect(Math.abs(v - fbm3(x, y, z + 0.01))).toBeLessThan(0.06);
    }
  });
});
