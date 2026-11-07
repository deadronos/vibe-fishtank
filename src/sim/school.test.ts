import { describe, expect, it } from "vitest";
import type { Vec3 } from "../lib/math";
import { TANK } from "../scene/common";
import { sandHeight } from "../scene/layout";
import { School, SPECIES, type Fish } from "./school";

const DT = 1 / 60;

function run(school: School, seconds: number, each?: () => void): void {
  for (let t = 0; t < seconds; t += DT) {
    school.update(DT);
    each?.();
  }
}

const distance = (f: Fish, p: Vec3) => Math.hypot(f.x - p[0], f.y - p[1], f.z - p[2]);
const mean = (values: number[]) => values.reduce((a, b) => a + b, 0) / values.length;

describe("School", () => {
  it("starts with every species at its configured count", () => {
    const school = new School();
    expect(school.fish).toHaveLength(SPECIES.reduce((sum, s) => sum + s.count, 0));
    SPECIES.forEach((def, i) => {
      expect(school.fish.filter((f) => f.species === i)).toHaveLength(def.count);
    });
  });

  it("is deterministic", () => {
    const a = new School();
    const b = new School();
    run(a, 2);
    run(b, 2);
    expect(a.fish).toEqual(b.fish);
  });

  it("keeps every fish inside the water, at a sane speed, with a valid orientation", () => {
    const school = new School();
    school.dropFood(0.4, 0);
    school.startle([TANK.halfX, 0.6, 0]);
    run(school, 20, () => {
      for (const f of school.fish) {
        expect(Math.abs(f.x)).toBeLessThanOrEqual(TANK.halfX);
        expect(Math.abs(f.z)).toBeLessThanOrEqual(TANK.halfZ);
        expect(f.y).toBeGreaterThanOrEqual(sandHeight(f.x, f.z));
        expect(f.y).toBeLessThanOrEqual(TANK.water);
        expect(Math.hypot(f.vx, f.vy, f.vz)).toBeLessThanOrEqual(f.def.maxSpeed + 1e-9);
      }
    });
    for (const f of school.fish) {
      expect(Math.hypot(f.fx, f.fy, f.fz)).toBeCloseTo(1);
      expect(Math.hypot(f.ux, f.uy, f.uz)).toBeCloseTo(1);
      // The up vector is kept perpendicular to the heading.
      expect(f.ux * f.fx + f.uy * f.fy + f.uz * f.fz).toBeCloseTo(0);
      expect(Number.isFinite(f.phase)).toBe(true);
    }
  });

  it("keeps bottom dwellers near the sand", () => {
    const school = new School();
    run(school, 10);
    const bottom = school.fish.filter((f) => f.def.bottom);
    expect(bottom.length).toBeGreaterThan(0);
    for (const f of bottom) expect(f.y - sandHeight(f.x, f.z)).toBeLessThan(0.15);
  });

  it("keeps schooling fish close to their own kind", () => {
    const school = new School();
    run(school, 15);
    const tetras = school.fish.filter((f) => f.species === 0);
    const cx = mean(tetras.map((f) => f.x));
    const cy = mean(tetras.map((f) => f.y));
    const cz = mean(tetras.map((f) => f.z));
    // The tank is 3 units wide; a school that has fallen apart would spread across most of it.
    expect(mean(tetras.map((f) => distance(f, [cx, cy, cz])))).toBeLessThan(0.6);
  });

  describe("food", () => {
    it("drops the requested amount at the surface, inside the tank", () => {
      const school = new School();
      school.dropFood(5, -5, 10);
      expect(school.food).toHaveLength(10);
      for (const f of school.food) {
        expect(Math.abs(f.x)).toBeLessThan(TANK.halfX);
        expect(Math.abs(f.z)).toBeLessThan(TANK.halfZ);
        expect(f.y).toBeCloseTo(TANK.water, 2);
      }
    });

    it("caps how much food can be in the water", () => {
      const school = new School();
      for (let i = 0; i < 20; i++) school.dropFood(0, 0, 10);
      expect(school.food).toHaveLength(80);
    });

    it("gets eaten", () => {
      const school = new School();
      school.dropFood(0, 0, 7);
      let fed = false;
      run(school, 15, () => {
        fed ||= school.fish.some((f) => f.fed > 0);
      });
      expect(fed).toBe(true);
      expect(school.food.length).toBeLessThan(7);
    });

    it("sinks to the sand, and is gone within a minute", () => {
      const school = new School();
      // Far corner, out of reach of most fish, so it has a chance to sink.
      school.dropFood(-1.4, 0.6, 3);
      run(school, 12);
      for (const f of school.food) expect(f.y).toBeLessThan(TANK.water - 0.5);
      run(school, 40);
      expect(school.food).toHaveLength(0);
    });
  });

  it("scatters fish away from a startle", () => {
    const point: Vec3 = [0, 0.7, 0];
    const calm = new School();
    const startled = new School();
    startled.startle(point);
    run(calm, 0.6);
    run(startled, 0.6);
    const near = (school: School) => mean(school.fish.map((f) => distance(f, point)));
    expect(near(startled)).toBeGreaterThan(near(calm) + 0.05);
  });
});
