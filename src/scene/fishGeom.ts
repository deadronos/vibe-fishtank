/** Fish geometry: each species' body plan, and the loft that turns a plan into a mesh. */

import { MeshBuilder } from "../lib/geometry";
import { mix, normalize, type Vec3 } from "../lib/math";

/** Fish meshes are one unit long, nose at x = 0.5 and tail tip at x = -0.5, swimming along +x. */
export interface Fin {
  from: number;
  to: number;
  height: number;
  sweep: number;
  /** Rises to a long point near the front, rather than a rounded edge. */
  pointed?: boolean;
}

export interface BodyPlan {
  halfHeight: number;
  halfWidth: number;
  /** Bigger pushes the deepest part of the body towards the nose. */
  bluntness: number;
  /** Fullness of the body outline. */
  fullness: number;
  bodyEnd: number;
  tail: { rootHeight: number; spread: number; fork: number };
  dorsal: Fin;
  anal: Fin;
  pectoral: number;
  /** Eye position in mesh space, and radius. */
  eye: Vec3;
  wave: number;
  amplitude: number;
}

export const PLANS: readonly BodyPlan[] = [
  // Neon tetra
  {
    halfHeight: 0.14,
    halfWidth: 0.055,
    bluntness: 0.72,
    fullness: 0.75,
    bodyEnd: -0.26,
    tail: { rootHeight: 0.03, spread: 0.15, fork: 0.1 },
    dorsal: { from: 0.1, to: -0.02, height: 0.1, sweep: 0.05 },
    anal: { from: 0.0, to: -0.2, height: 0.07, sweep: 0.04 },
    pectoral: 0.09,
    eye: [0.355, 0.03, 0.03],
    wave: 7,
    amplitude: 0.1,
  },
  // Ember rasbora
  {
    halfHeight: 0.15,
    halfWidth: 0.06,
    bluntness: 0.68,
    fullness: 0.7,
    bodyEnd: -0.27,
    tail: { rootHeight: 0.03, spread: 0.16, fork: 0.12 },
    dorsal: { from: 0.04, to: -0.08, height: 0.09, sweep: 0.05 },
    anal: { from: -0.02, to: -0.17, height: 0.07, sweep: 0.03 },
    pectoral: 0.08,
    eye: [0.36, 0.035, 0.034],
    wave: 7,
    amplitude: 0.11,
  },
  // Angelfish: deep body with long swept fins
  {
    halfHeight: 0.33,
    halfWidth: 0.05,
    bluntness: 0.85,
    fullness: 0.62,
    bodyEnd: -0.17,
    tail: { rootHeight: 0.05, spread: 0.2, fork: -0.04 },
    dorsal: { from: 0.14, to: -0.15, height: 0.36, sweep: 0.52, pointed: true },
    anal: { from: 0.1, to: -0.15, height: 0.4, sweep: 0.56, pointed: true },
    pectoral: 0.08,
    eye: [0.3, 0.05, 0.022],
    wave: 5,
    amplitude: 0.05,
  },
  // Discus: nearly round
  {
    halfHeight: 0.36,
    halfWidth: 0.065,
    bluntness: 0.95,
    fullness: 0.5,
    bodyEnd: -0.3,
    tail: { rootHeight: 0.05, spread: 0.13, fork: -0.02 },
    dorsal: { from: 0.2, to: -0.3, height: 0.07, sweep: 0.03 },
    anal: { from: 0.1, to: -0.3, height: 0.07, sweep: 0.03 },
    pectoral: 0.08,
    eye: [0.3, 0.07, 0.03],
    wave: 5,
    amplitude: 0.05,
  },
  // Corydoras: stout and low
  {
    halfHeight: 0.15,
    halfWidth: 0.095,
    bluntness: 0.82,
    fullness: 0.56,
    bodyEnd: -0.27,
    tail: { rootHeight: 0.035, spread: 0.13, fork: 0.07 },
    dorsal: { from: 0.18, to: 0.02, height: 0.14, sweep: 0.06 },
    anal: { from: -0.05, to: -0.16, height: 0.06, sweep: 0.03 },
    pectoral: 0.11,
    eye: [0.34, 0.05, 0.03],
    wave: 7,
    amplitude: 0.08,
  },
];

/** The outline of a dorsal or anal fin, from its leading edge to its trailing edge. */
const finShape = (s: number, pointed: boolean): number => {
  if (pointed)
    return s < 0.22
      ? Math.pow(s / 0.22, 1.3)
      : Math.pow((1 - s) / 0.78, 1.5) * 0.92 + 0.08 * (1 - s);
  return Math.pow(Math.sin(Math.PI * Math.min(1, s * 0.82 + 0.1)), 0.7);
};

export function buildFish(plan: BodyPlan): MeshBuilder {
  const b = new MeshBuilder();
  const NOSE = 0.5;
  const bodyLength = NOSE - plan.bodyEnd;

  const outline = (x: number): number => {
    const s = Math.min(1, Math.max(0, (NOSE - x) / bodyLength));
    const body = Math.pow(Math.sin(Math.PI * Math.pow(s, plan.bluntness)), plan.fullness);
    return Math.max(body, 0.1 * s);
  };

  b.grid(16, 28, (around, along) => {
    const x = NOSE - along * bodyLength;
    const profile = outline(x);
    const theta = around * Math.PI * 2;
    const h = plan.halfHeight * profile;
    const w = plan.halfWidth * Math.pow(profile, 0.8);
    const [c, s] = [Math.cos(theta), Math.sin(theta)];
    const nearNose = along < 0.12 ? (0.12 - along) * 5 : 0;
    return {
      p: [x, s * h, c * w],
      n: normalize([
        nearNose,
        s / Math.max(plan.halfHeight, 0.01),
        c / Math.max(plan.halfWidth, 0.01),
      ]),
      u: NOSE - x,
      v: s,
      extra: 0,
    };
  });

  const median = (root: (s: number) => Vec3, tip: (s: number) => Vec3, columns: number): void => {
    b.grid(columns, 4, (s, t) => {
      const [r, e] = [root(s), tip(s)];
      const p: Vec3 = [mix(r[0], e[0], t), mix(r[1], e[1], t), mix(r[2], e[2], t)];
      return { p, n: [0, 0, 1], u: NOSE - p[0], v: t, extra: Math.max(t, 0.02) };
    });
  };

  const { tail, dorsal, anal } = plan;
  const tailStart = plan.bodyEnd + 0.03;
  median(
    (s) => [tailStart, (s * 2 - 1) * tail.rootHeight, 0],
    (s) => {
      const q = s * 2 - 1;
      return [-0.5 + tail.fork * (1 - Math.abs(q)), q * tail.spread, 0];
    },
    12,
  );

  const edgeFin = (fin: Fin, sign: number): void => {
    median(
      (s) => {
        const x = mix(fin.from, fin.to, s);
        return [x, sign * plan.halfHeight * outline(x) * 0.93, 0];
      },
      (s) => {
        const x = mix(fin.from, fin.to, s);
        return [
          x - fin.sweep * (0.25 + s),
          sign *
            (plan.halfHeight * outline(x) * 0.93 + fin.height * finShape(s, fin.pointed ?? false)),
          0,
        ];
      },
      10,
    );
  };
  edgeFin(dorsal, 1);
  edgeFin(anal, -1);

  for (const side of [1, -1]) {
    const size = plan.pectoral;
    const y = -plan.halfHeight * 0.32;
    const z = side * plan.halfWidth * 0.82;
    b.grid(4, 3, (s, t) => {
      const x = 0.26 - s * size * 0.45;
      const p: Vec3 = [
        x - t * size * 0.9,
        y - t * size * 0.55,
        z + side * t * size * (0.5 + s * 0.2),
      ];
      return { p, n: normalize([0, 0.4, side]), u: NOSE - p[0], v: t, extra: Math.max(t, 0.02) };
    });
  }
  return b;
}
