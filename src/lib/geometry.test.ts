import { describe, expect, it } from "vitest";
import { icosphere, MeshBuilder } from "./geometry";

describe("MeshBuilder.grid", () => {
  it("builds (nu+1)×(nv+1) vertices and 6 indices per cell", () => {
    const b = new MeshBuilder();
    b.grid(2, 2, (s) => ({ p: [s, 0, 0], n: [0, 1, 0] }));
    const g = b.build();
    expect(g.positions.length / 3).toBe(9);
    expect(g.normals.length / 3).toBe(9);
    expect(g.uvs.length / 2).toBe(9);
    expect(g.extras.length).toBe(9);
    expect(g.indices.length).toBe(24);
    expect(Math.max(...g.indices)).toBeLessThan(9);
    expect(Math.min(...g.indices)).toBe(0);
  });

  it("passes normalized coordinates to the callback, row by row", () => {
    const b = new MeshBuilder();
    const seen: Array<[number, number]> = [];
    b.grid(2, 1, (s, t) => {
      seen.push([s, t]);
      return { p: [0, 0, 0], n: [0, 1, 0] };
    });
    expect(seen[0]).toEqual([0, 0]);
    expect(seen[1]).toEqual([0.5, 0]);
    expect(seen[2]).toEqual([1, 0]);
    expect(seen[3]).toEqual([0, 1]);
    expect(seen[5]).toEqual([1, 1]);
  });
});

describe("MeshBuilder.quad and box", () => {
  it("stores the given normal on all four corners", () => {
    const b = new MeshBuilder();
    b.quad([0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1]);
    const g = b.build();
    expect(g.positions.length / 3).toBe(4);
    expect(g.indices).toEqual(new Uint32Array([0, 1, 2, 0, 2, 3]));
    for (let i = 0; i < 4; i++)
      expect(Array.from(g.normals.slice(i * 3, i * 3 + 3))).toEqual([0, 0, 1]);
  });

  it("builds a closed box with axis-aligned unit normals", () => {
    const b = new MeshBuilder();
    b.box([-1, -2, -3], [1, 2, 3]);
    const g = b.build();
    expect(g.positions.length / 3).toBe(24);
    expect(g.indices.length).toBe(36);
    for (let i = 0; i < 24; i++) {
      const n = Array.from(g.normals.slice(i * 3, i * 3 + 3));
      const absolute = n.map(Math.abs);
      expect(absolute[0] + absolute[1] + absolute[2]).toBeCloseTo(1, 12);
      for (const component of n) {
        expect(Math.abs(component) === 0 || Math.abs(component) === 1).toBe(true);
      }
      // Every vertex sits on the box shell.
      const p = Array.from(g.positions.slice(i * 3, i * 3 + 3));
      expect(p[0]).toBeCloseTo(Math.sign(p[0]), 12);
      expect(p[1]).toBeCloseTo(Math.sign(p[1]) * 2, 12);
      expect(p[2]).toBeCloseTo(Math.sign(p[2]) * 3, 12);
    }
  });
});

describe("MeshBuilder.smooth", () => {
  it("gives every vertex of a flat triangle its face normal", () => {
    const b = new MeshBuilder();
    b.smooth(
      [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
      ],
      [0, 1, 2],
    );
    const g = b.build();
    expect(g.positions.length / 3).toBe(3);
    for (let i = 0; i < 3; i++) {
      const n = Array.from(g.normals.slice(i * 3, i * 3 + 3));
      expect(n[0]).toBeCloseTo(0, 12);
      expect(n[1]).toBeCloseTo(0, 12);
      expect(n[2]).toBeCloseTo(1, 12);
    }
  });
});

describe("MeshBuilder.tube", () => {
  it("sweeps rings of the given radius around the spine", () => {
    const b = new MeshBuilder();
    b.tube(
      [
        [0, 0, 0],
        [0, 1, 0],
      ],
      [0.5, 0.25],
      4,
    );
    const g = b.build();
    expect(g.positions.length / 3).toBe(10); // (sides+1) rings × 2 points
    expect(g.indices.length).toBe(24);
    for (let i = 0; i < 10; i++) {
      const p = Array.from(g.positions.slice(i * 3, i * 3 + 3));
      const ring = i < 5 ? 0 : 1;
      // Vertices pass through float32 storage, so a float32-epsilon tolerance.
      expect(Math.hypot(p[0], p[2])).toBeCloseTo(ring === 0 ? 0.5 : 0.25, 6);
      expect(p[1]).toBeCloseTo(ring, 6);
      const n = Array.from(g.normals.slice(i * 3, i * 3 + 3));
      expect(Math.hypot(n[0], n[1], n[2])).toBeCloseTo(1, 6);
    }
  });
});

const edgeCounts = (indices: number[]): Map<number, number> => {
  const edges = new Map<number, number>();
  for (let i = 0; i < indices.length; i += 3) {
    for (const [a, b] of [
      [indices[i], indices[i + 1]],
      [indices[i + 1], indices[i + 2]],
      [indices[i + 2], indices[i]],
    ]) {
      const key = Math.min(a, b) * 1000000 + Math.max(a, b);
      edges.set(key, (edges.get(key) ?? 0) + 1);
    }
  }
  return edges;
};

describe("icosphere", () => {
  it("subdivides a closed, manifold, unit sphere", () => {
    for (const subdivisions of [0, 1, 2]) {
      const { positions, indices } = icosphere(subdivisions);
      const vertices = positions.length;
      const faces = indices.length / 3;
      const edges = edgeCounts(indices);
      // Euler characteristic of a sphere.
      expect(vertices - edges.size + faces).toBe(2);
      // Every edge is shared by exactly two faces: the mesh has no cracks.
      for (const count of edges.values()) expect(count).toBe(2);
      // All vertices on the unit sphere, all indices valid, none repeated.
      const seen = new Set<string>();
      for (const p of positions) {
        expect(Math.hypot(p[0], p[1], p[2])).toBeCloseTo(1, 6);
        seen.add(p.map((v) => v.toFixed(6)).join(","));
      }
      expect(seen.size).toBe(vertices);
      expect(Math.max(...indices)).toBe(vertices - 1);
      expect(Math.min(...indices)).toBe(0);
    }
  });

  it("quadruples the face count per subdivision level", () => {
    expect(icosphere(0).indices.length).toBe(60);
    expect(icosphere(1).indices.length).toBe(240);
    expect(icosphere(2).indices.length).toBe(960);
  });
});
