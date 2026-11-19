import { describe, expect, it } from "vitest";
import { TANK } from "./common";
import { createObstacles, DRIFTWOOD, sandHeight, STONES, stoneClearance } from "./layout";

const grid = (n: number): [number, number][] => {
  const points: [number, number][] = [];
  for (let i = 0; i <= n; i++) {
    for (let j = 0; j <= n; j++) {
      points.push([(i / n - 0.5) * 2 * TANK.halfX, (j / n - 0.5) * 2 * TANK.halfZ]);
    }
  }
  return points;
};

/** Summed sand height along a line across the tank at depth `z`. */
const row = (z: number) => {
  let sum = 0;
  for (let x = -1.4; x <= 1.4; x += 0.1) sum += sandHeight(x, z);
  return sum;
};

describe("sandHeight", () => {
  it("stays well below the water across the whole tank", () => {
    for (const [x, z] of grid(30)) {
      const h = sandHeight(x, z);
      expect(h).toBeGreaterThan(0.05);
      expect(h).toBeLessThan(0.4);
    }
  });

  it("slopes up towards the back", () => {
    expect(row(-TANK.halfZ + 0.05)).toBeGreaterThan(row(TANK.halfZ - 0.05));
  });

  it("mounds up around the stones", () => {
    const stone = STONES[0];
    expect(sandHeight(stone.x, stone.z)).toBeGreaterThan(sandHeight(stone.x, stone.z + 0.6));
  });
});

describe("stoneClearance", () => {
  it("is 0 at a stone and 1 in open sand", () => {
    for (const stone of STONES) expect(stoneClearance(stone.x, stone.z)).toBe(0);
    expect(stoneClearance(0, 0.6)).toBe(1);
  });

  it("stays within 0..1", () => {
    for (const [x, z] of grid(20)) {
      const c = stoneClearance(x, z);
      expect(c).toBeGreaterThanOrEqual(0);
      expect(c).toBeLessThanOrEqual(1);
    }
  });
});

describe("createObstacles", () => {
  it("has one obstacle per stone and per driftwood point", () => {
    const points = DRIFTWOOD.reduce((sum, branch) => sum + branch.length, 0);
    expect(createObstacles()).toHaveLength(STONES.length + points);
  });

  it("sits stone obstacles on the sand", () => {
    createObstacles()
      .slice(0, STONES.length)
      .forEach((obstacle, i) => {
        const stone = STONES[i];
        expect(obstacle.center[1]).toBeGreaterThan(sandHeight(stone.x, stone.z));
        expect(obstacle.radius).toBeGreaterThanOrEqual(stone.radius);
      });
  });
});
