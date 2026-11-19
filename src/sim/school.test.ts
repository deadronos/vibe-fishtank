import { describe, expect, it } from "vitest";
import type { Vec3 } from "../lib/math";
import { TANK } from "../scene/common";
import { sandHeight } from "../scene/layout";
import { School, SPECIES, type Fish, type Food } from "./school";

const DT = 1 / 60;

const run = (school: School, seconds: number): void => {
  const steps = Math.round(seconds / DT);
  for (let i = 0; i < steps; i++) school.update(DT);
};

const speed = (fish: Fish): number => Math.hypot(fish.vx, fish.vy, fish.vz);

const distance = (fish: Fish, point: readonly number[]): number =>
  Math.hypot(fish.x - point[0], fish.y - point[1], fish.z - point[2]);

const mean = (values: readonly number[]): number =>
  values.reduce((sum, v) => sum + v, 0) / values.length;

describe("construction", () => {
  it("creates every fish from the species table, starting inside the tank", () => {
    const school = new School();
    expect(school.fish.length).toBe(SPECIES.reduce((sum, def) => sum + def.count, 0));
    SPECIES.forEach((def, species) => {
      const group = school.fish.filter((f) => f.species === species);
      expect(group.length).toBe(def.count);
      for (const fish of group) {
        expect(Math.abs(fish.x)).toBeLessThan(TANK.halfX);
        expect(Math.abs(fish.z)).toBeLessThan(TANK.halfZ);
        expect(fish.y).toBeGreaterThan(0);
        expect(fish.y).toBeLessThan(TANK.water);
      }
    });
  });
});

const snapshot = (school: School): number[] =>
  school.fish.flatMap((f) => [f.x, f.y, f.z, f.fx, f.fy, f.fz, f.phase]);

describe("invariants under long runs", () => {
  it("keeps every fish inside the tank and above the sand, even while startled", () => {
    const school = new School();
    // 60 simulated seconds, with glass taps mixed in so fish hit the walls fast.
    for (let step = 0; step < 3600; step++) {
      if (step % 600 === 300) school.startle([0, 0.7, TANK.halfZ]);
      school.update(DT);
      if (step % 5 === 0) {
        for (const fish of school.fish) {
          expect(Math.abs(fish.x)).toBeLessThan(TANK.halfX);
          expect(Math.abs(fish.z)).toBeLessThan(TANK.halfZ);
          expect(fish.y).toBeGreaterThan(sandHeight(fish.x, fish.z) - 1e-9);
          expect(fish.y).toBeLessThan(TANK.water);
        }
      }
    }
  });

  it("keeps headings and up-vectors orthonormal and speeds within species limits", () => {
    const school = new School();
    run(school, 5);
    for (const fish of school.fish) {
      expect(Math.hypot(fish.fx, fish.fy, fish.fz)).toBeCloseTo(1, 6);
      expect(Math.hypot(fish.ux, fish.uy, fish.uz)).toBeCloseTo(1, 6);
      expect(fish.fx * fish.ux + fish.fy * fish.uy + fish.fz * fish.uz).toBeCloseTo(0, 8);
      expect(fish.turn).toBeGreaterThanOrEqual(-1);
      expect(fish.turn).toBeLessThanOrEqual(1);
      expect(speed(fish)).toBeGreaterThan(0);
      expect(speed(fish)).toBeLessThanOrEqual(fish.def.maxSpeed + 1e-6);
    }
  });

  it("is deterministic for a given seed", () => {
    const a = new School();
    const b = new School();
    a.dropFood(0.2, 0.1, 5);
    b.dropFood(0.2, 0.1, 5);
    const steps = Math.round(10 / DT);
    for (let i = 0; i < steps; i++) {
      a.update(DT);
      b.update(DT);
    }
    expect(snapshot(b)).toEqual(snapshot(a));
    expect(b.food.length).toBe(a.food.length);
  });
});

describe("startle", () => {
  it("bursts nearby fish away from the tap", () => {
    const school = new School();
    run(school, 2);
    const point: Vec3 = [0, 0.7, TANK.halfZ];
    const affected = school.fish.filter((f) => distance(f, point) < 1.6);
    expect(affected.length).toBeGreaterThan(10);
    const distanceBefore = mean(affected.map((f) => distance(f, point)));
    const speedBefore = mean(affected.map(speed));

    school.startle(point);
    run(school, 0.5);

    const distanceAfter = mean(affected.map((f) => distance(f, point)));
    const speedAfter = mean(affected.map(speed));
    expect(distanceAfter).toBeGreaterThan(distanceBefore);
    expect(speedAfter).toBeGreaterThan(speedBefore + 0.05);
  });
});

describe("food", () => {
  it("gets eaten before it can expire, and the tank clears it entirely", () => {
    const school = new School();
    run(school, 1);
    school.dropFood(0.3, -0.1, 80);
    expect(school.food.length).toBe(80);
    // After 5 seconds nothing has expired yet (food lives 45s), so any
    // missing flakes must have been eaten.
    run(school, 5);
    expect(school.food.length).toBeLessThan(80);
    // Everything is gone well before the 45-second expiry.
    run(school, 45);
    expect(school.food.length).toBe(0);
  });

  it("food only ever sinks, never rises, and never penetrates the substrate", () => {
    const school = new School();
    run(school, 1);
    school.dropFood(0.5, 0.1, 80);
    const lastY = new Map<Food, number>();
    for (let step = 0; step < 480; step++) {
      school.update(DT);
      for (const flake of school.food) {
        const previous = lastY.get(flake);
        if (previous !== undefined) {
          expect(flake.y).toBeLessThanOrEqual(previous + 1e-12);
        }
        lastY.set(flake, flake.y);
        expect(flake.y).toBeGreaterThanOrEqual(sandHeight(flake.x, flake.z) + 0.006 - 1e-9);
        expect(Math.abs(flake.x)).toBeLessThan(TANK.halfX);
        expect(Math.abs(flake.z)).toBeLessThan(TANK.halfZ);
      }
    }
  });
});

describe("bottom dwellers", () => {
  it("corydoras hug the sand rather than the water column", () => {
    const school = new School();
    run(school, 30);
    const bottomDwellers = school.fish.filter((f) => f.def.bottom);
    expect(bottomDwellers.length).toBeGreaterThan(0);
    for (const fish of bottomDwellers) {
      const floor = sandHeight(fish.x, fish.z);
      expect(fish.y).toBeGreaterThan(floor - 1e-6);
      expect(fish.y).toBeLessThan(floor + 0.5);
    }
    // They live clearly below the mid-column species.
    const others = school.fish.filter((f) => !f.def.bottom);
    expect(mean(bottomDwellers.map((f) => f.y))).toBeLessThan(mean(others.map((f) => f.y)));
  });
});
