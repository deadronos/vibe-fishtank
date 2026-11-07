import { describe, expect, it } from "vitest";
import type { Geometry } from "./gpu/types";
import { icosphere, MeshBuilder } from "./geometry";
import { cross, dot, length, sub, type Vec3 } from "./math";

const vertexCount = (g: Geometry): number => g.positions.length / 3;
const at = (array: Float32Array, i: number): Vec3 => [
  array[i * 3],
  array[i * 3 + 1],
  array[i * 3 + 2],
];

/** Every index in range, and every attribute sized for the same number of vertices. */
function expectConsistent(g: Geometry): void {
  const n = vertexCount(g);
  expect(g.normals.length).toBe(n * 3);
  expect(g.uvs.length).toBe(n * 2);
  expect(g.extras.length).toBe(n);
  expect(g.indices.length % 3).toBe(0);
  for (const index of g.indices) expect(index).toBeLessThan(n);
}

/** The normal implied by each triangle's winding, which must be counter-clockwise from outside. */
function faceNormals(g: Geometry): Vec3[] {
  const normals: Vec3[] = [];
  for (let i = 0; i < g.indices.length; i += 3) {
    const [a, b, c] = [g.indices[i], g.indices[i + 1], g.indices[i + 2]].map((v) =>
      at(g.positions, v),
    );
    normals.push(cross(sub(b, a), sub(c, a)));
  }
  return normals;
}

describe("MeshBuilder", () => {
  it("numbers vertices in order and keeps their attributes", () => {
    const mesh = new MeshBuilder();
    expect(mesh.vertex([0, 0, 0], [0, 1, 0], 0.25, 0.5, 3)).toBe(0);
    expect(mesh.vertex([1, 0, 0], [0, 1, 0])).toBe(1);
    mesh.triangle(0, 1, 0);
    const g = mesh.build();
    expectConsistent(g);
    expect([...g.uvs]).toEqual([0.25, 0.5, 0, 0]);
    expect([...g.extras]).toEqual([3, 0]);
    expect([...g.indices]).toEqual([0, 1, 0]);
  });

  it("builds a grid with the expected counts and default uvs", () => {
    const mesh = new MeshBuilder();
    mesh.vertex([9, 9, 9], [0, 1, 0]); // The grid must offset its indices past this.
    mesh.grid(4, 3, (s, t) => ({ p: [s, 0, t], n: [0, 1, 0] }));
    const g = mesh.build();
    expectConsistent(g);
    expect(vertexCount(g)).toBe(1 + 5 * 4);
    expect(g.indices.length).toBe(4 * 3 * 6);
    expect(Math.min(...g.indices)).toBe(1);
    expect([g.uvs[g.uvs.length - 2], g.uvs[g.uvs.length - 1]]).toEqual([1, 1]);
  });

  it("winds every box face outwards, matching its normal", () => {
    const mesh = new MeshBuilder();
    mesh.box([-1, -2, -3], [1, 2, 3], 5);
    const g = mesh.build();
    expectConsistent(g);
    expect(vertexCount(g)).toBe(24);
    expect(g.indices.length).toBe(36);
    faceNormals(g).forEach((face, i) => {
      const vertex = g.indices[i * 3];
      const normal = at(g.normals, vertex);
      expect(dot(face, normal)).toBeGreaterThan(0);
      // The box is centered on the origin, so outwards means away from it.
      expect(dot(normal, at(g.positions, vertex))).toBeGreaterThan(0);
    });
    expect(g.extras.every((e) => e === 5)).toBe(true);
  });

  it("computes smooth normals that point out of a closed sphere", () => {
    const sphere = icosphere(2);
    const mesh = new MeshBuilder();
    mesh.smooth(sphere.positions, sphere.indices);
    const g = mesh.build();
    expectConsistent(g);
    for (let i = 0; i < vertexCount(g); i++) {
      const n = at(g.normals, i);
      expect(length(n)).toBeCloseTo(1);
      // On a unit sphere the smooth normal is close to the position itself.
      expect(dot(n, at(g.positions, i))).toBeGreaterThan(0.99);
    }
  });

  it("sweeps a tube at the given radius around each point", () => {
    const points: Vec3[] = [
      [0, 0, 0],
      [0, 1, 0],
      [0.5, 2, 0.2],
    ];
    const radii = [0.3, 0.2, 0.1];
    const sides = 8;
    const mesh = new MeshBuilder();
    mesh.tube(points, radii, sides, 2);
    const g = mesh.build();
    expectConsistent(g);
    expect(vertexCount(g)).toBe((sides + 1) * points.length);
    for (let ring = 0; ring < points.length; ring++) {
      for (let side = 0; side <= sides; side++) {
        const p = at(g.positions, ring * (sides + 1) + side);
        expect(length(sub(p, points[ring]))).toBeCloseTo(radii[ring]);
      }
    }
    // v runs along the tube's length, so it increases ring by ring.
    const v = (ring: number) => g.uvs[ring * (sides + 1) * 2 + 1];
    expect(v(0)).toBe(0);
    expect(v(1)).toBeCloseTo(1);
    expect(v(2)).toBeGreaterThan(v(1));
  });
});

describe("icosphere", () => {
  it.each([0, 1, 2, 3])("has the expected counts at subdivision %i", (level) => {
    const { positions, indices } = icosphere(level);
    const faces = 20 * 4 ** level;
    expect(indices.length).toBe(faces * 3);
    // Shared midpoints: V = F / 2 + 2 for a closed triangle mesh of genus zero.
    expect(positions.length).toBe(faces / 2 + 2);
  });

  it("puts every vertex on the unit sphere", () => {
    for (const p of icosphere(2).positions) expect(length(p)).toBeCloseTo(1);
  });
});
