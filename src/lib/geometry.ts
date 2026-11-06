/** Procedural mesh construction. */

import type { Geometry } from "./gl";
import { cross, length, normalize, sub, type Vec3 } from "./math";

export interface GridVertex {
  p: Vec3;
  n: Vec3;
  u?: number;
  v?: number;
  extra?: number;
}

export class MeshBuilder {
  private readonly positions: number[] = [];
  private readonly normals: number[] = [];
  private readonly uvs: number[] = [];
  private readonly extras: number[] = [];
  private readonly indices: number[] = [];

  vertex(p: Vec3, n: Vec3, u = 0, v = 0, extra = 0): number {
    const index = this.positions.length / 3;
    this.positions.push(p[0], p[1], p[2]);
    this.normals.push(n[0], n[1], n[2]);
    this.uvs.push(u, v);
    this.extras.push(extra);
    return index;
  }

  triangle(a: number, b: number, c: number): void {
    this.indices.push(a, b, c);
  }

  /** A (nu + 1) x (nv + 1) vertex grid. `fn` receives normalized coordinates. */
  grid(nu: number, nv: number, fn: (s: number, t: number) => GridVertex): void {
    const base = this.positions.length / 3;
    for (let j = 0; j <= nv; j++) {
      for (let i = 0; i <= nu; i++) {
        const { p, n, u, v, extra } = fn(i / nu, j / nv);
        this.vertex(p, n, u ?? i / nu, v ?? j / nv, extra ?? 0);
      }
    }
    const row = nu + 1;
    for (let j = 0; j < nv; j++) {
      for (let i = 0; i < nu; i++) {
        const a = base + j * row + i;
        this.indices.push(a, a + 1, a + row, a + 1, a + row + 1, a + row);
      }
    }
  }

  /** A flat quad from four corners in order. */
  quad(a: Vec3, b: Vec3, c: Vec3, d: Vec3, n: Vec3, extra = 0): void {
    const base = this.vertex(a, n, 0, 0, extra);
    this.vertex(b, n, 1, 0, extra);
    this.vertex(c, n, 1, 1, extra);
    this.vertex(d, n, 0, 1, extra);
    this.indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
  }

  box(min: Vec3, max: Vec3, extra = 0): void {
    const [x0, y0, z0] = min;
    const [x1, y1, z1] = max;
    this.quad([x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1], [0, 0, 1], extra);
    this.quad([x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0], [0, 0, -1], extra);
    this.quad([x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [1, 0, 0], extra);
    this.quad([x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0], [-1, 0, 0], extra);
    this.quad([x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0], [0, 1, 0], extra);
    this.quad([x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1], [0, -1, 0], extra);
  }

  /** Adds raw triangles with smooth normals computed from the faces. */
  smooth(positions: readonly Vec3[], indices: readonly number[], extra = 0): void {
    const accumulated = positions.map((): [number, number, number] => [0, 0, 0]);
    for (let i = 0; i < indices.length; i += 3) {
      const [a, b, c] = [indices[i], indices[i + 1], indices[i + 2]];
      const face = cross(sub(positions[b], positions[a]), sub(positions[c], positions[a]));
      for (const index of [a, b, c]) {
        accumulated[index][0] += face[0];
        accumulated[index][1] += face[1];
        accumulated[index][2] += face[2];
      }
    }
    const base = this.positions.length / 3;
    positions.forEach((p, i) => this.vertex(p, normalize(accumulated[i]), 0, 0, extra));
    for (const index of indices) this.indices.push(base + index);
  }

  /** A tube swept along a polyline, with a radius given per point. */
  tube(points: readonly Vec3[], radii: readonly number[], sides: number, extra = 0): void {
    const count = points.length;
    let reference: Vec3 = [0.3, 0.2, 1];
    let along = 0;
    this.grid(sides, count - 1, (s, t) => {
      const index = Math.min(count - 1, Math.round(t * (count - 1)));
      const prev = points[Math.max(0, index - 1)];
      const next = points[Math.min(count - 1, index + 1)];
      const tangent = normalize(sub(next, prev));
      const side = normalize(cross(tangent, reference));
      const up = cross(side, tangent);
      if (s === 0) {
        reference = up;
        if (index > 0) along += length(sub(points[index], points[index - 1]));
      }
      const angle = s * Math.PI * 2;
      const n = normalize([
        side[0] * Math.cos(angle) + up[0] * Math.sin(angle),
        side[1] * Math.cos(angle) + up[1] * Math.sin(angle),
        side[2] * Math.cos(angle) + up[2] * Math.sin(angle),
      ]);
      const r = radii[index];
      return {
        p: [points[index][0] + n[0] * r, points[index][1] + n[1] * r, points[index][2] + n[2] * r],
        n,
        u: s,
        v: along,
        extra,
      };
    });
  }

  build(): Geometry {
    return {
      positions: new Float32Array(this.positions),
      normals: new Float32Array(this.normals),
      uvs: new Float32Array(this.uvs),
      extras: new Float32Array(this.extras),
      indices: new Uint32Array(this.indices),
    };
  }
}

/** A unit icosphere with shared vertices. */
export function icosphere(subdivisions: number): { positions: Vec3[]; indices: number[] } {
  const g = (1 + Math.sqrt(5)) / 2;
  const positions: Vec3[] = (
    [
      [-1, g, 0],
      [1, g, 0],
      [-1, -g, 0],
      [1, -g, 0],
      [0, -1, g],
      [0, 1, g],
      [0, -1, -g],
      [0, 1, -g],
      [g, 0, -1],
      [g, 0, 1],
      [-g, 0, -1],
      [-g, 0, 1],
    ] as Vec3[]
  ).map(normalize);
  let indices = [
    0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11, 1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1,
    8, 3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9, 4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1,
  ];
  for (let level = 0; level < subdivisions; level++) {
    const midpoints = new Map<number, number>();
    const midpoint = (a: number, b: number): number => {
      const key = Math.min(a, b) * 1_000_000 + Math.max(a, b);
      let index = midpoints.get(key);
      if (index === undefined) {
        const [pa, pb] = [positions[a], positions[b]];
        index = positions.length;
        positions.push(normalize([pa[0] + pb[0], pa[1] + pb[1], pa[2] + pb[2]]));
        midpoints.set(key, index);
      }
      return index;
    };
    const next: number[] = [];
    for (let i = 0; i < indices.length; i += 3) {
      const [a, b, c] = [indices[i], indices[i + 1], indices[i + 2]];
      const [ab, bc, ca] = [midpoint(a, b), midpoint(b, c), midpoint(c, a)];
      next.push(a, ab, ca, b, bc, ab, c, ca, bc, ab, bc, ca);
    }
    indices = next;
  }
  return { positions, indices };
}
