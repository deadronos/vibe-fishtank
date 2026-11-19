import { describe, expect, it } from "vitest";
import { TANK } from "./common";
import { createObstacles, DRIFTWOOD, sandHeight, STONES, stoneClearance } from "./layout";

describe("sandHeight", () => {
  it("keeps the substrate inside plausible bounds across the whole tank", () => {
    for (let i = 0; i <= 40; i++) {
      for (let j = 0; j <= 20; j++) {
        const x = (i / 40) * 2 * TANK.halfX - TANK.halfX;
        const z = (j / 20) * 2 * TANK.halfZ - TANK.halfZ;
        const h = sandHeight(x, z);
        expect(h).toBeGreaterThan(0);
        expect(h).toBeLessThan(TANK.height * 0.3);
      }
    }
  });

  it("slopes up towards the back at every column", () => {
    for (let i = 0; i <= 30; i++) {
      const x = (i / 30) * 2 * (TANK.halfX - 0.05) - (TANK.halfX - 0.05);
      const back = sandHeight(x, -TANK.halfZ + 0.03);
      const front = sandHeight(x, TANK.halfZ - 0.03);
      expect(back).toBeGreaterThan(front);
    }
  });

  it("is deterministic", () => {
    for (const [x, z] of [
      [0, 0],
      [0.7, -0.3],
      [-1.2, 0.4],
    ]) {
      expect(sandHeight(x, z)).toBe(sandHeight(x, z));
    }
  });
});

describe("stoneClearance", () => {
  it("is zero at a stone's centre and one far away", () => {
    for (const stone of STONES) {
      expect(stoneClearance(stone.x, stone.z)).toBe(0);
    }
    expect(stoneClearance(TANK.halfX * 0.9, TANK.halfZ * 0.9)).toBe(1);
  });

  it("stays within 0..1 everywhere", () => {
    for (let i = 0; i <= 40; i++) {
      for (let j = 0; j <= 20; j++) {
        const x = (i / 40) * 2 * TANK.halfX - TANK.halfX;
        const z = (j / 20) * 2 * TANK.halfZ - TANK.halfZ;
        const c = stoneClearance(x, z);
        expect(c).toBeGreaterThanOrEqual(0);
        expect(c).toBeLessThanOrEqual(1);
      }
    }
  });
});

describe("createObstacles", () => {
  it("covers every stone and every driftwood knot", () => {
    const obstacles = createObstacles();
    const woodPoints = DRIFTWOOD.reduce((sum, branch) => sum + branch.length, 0);
    expect(obstacles.length).toBe(STONES.length + woodPoints);
    for (const obstacle of obstacles) {
      expect(obstacle.radius).toBeGreaterThan(0);
      expect(Math.abs(obstacle.center[0])).toBeLessThanOrEqual(TANK.halfX);
      expect(Math.abs(obstacle.center[2])).toBeLessThanOrEqual(TANK.halfZ);
      expect(obstacle.center[1]).toBeGreaterThan(0);
      expect(obstacle.center[1]).toBeLessThan(TANK.height);
    }
    // Stone obstacles sit on the substrate they belong to.
    for (let i = 0; i < STONES.length; i++) {
      const stone = STONES[i];
      const center = obstacles[i].center;
      expect(center[0]).toBeCloseTo(stone.x, 12);
      expect(center[2]).toBeCloseTo(stone.z, 12);
      expect(center[1]).toBeGreaterThan(sandHeight(stone.x, stone.z));
    }
  });
});
