/** Tiny linear algebra helpers. Matrices are column-major, as GLSL expects. */

export type Floats = Float32Array<ArrayBuffer>;
export type Vec3 = readonly [number, number, number];

export const TAU = Math.PI * 2;

export const clamp = (x: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, x));
export const mix = (a: number, b: number, t: number): number => a + (b - a) * t;
export const smoothstep = (a: number, b: number, x: number): number => {
  const t = clamp((x - a) / (b - a), 0, 1);
  return t * t * (3 - 2 * t);
};

export const add = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const scale = (a: Vec3, s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s];
export const dot = (a: Vec3, b: Vec3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross = (a: Vec3, b: Vec3): Vec3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
export const length = (a: Vec3): number => Math.hypot(a[0], a[1], a[2]);
export const normalize = (a: Vec3): Vec3 => {
  const l = length(a) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
};
export const lerp3 = (a: Vec3, b: Vec3, t: number): Vec3 => [
  mix(a[0], b[0], t),
  mix(a[1], b[1], t),
  mix(a[2], b[2], t),
];

export function perspective(fovY: number, aspect: number, near: number, far: number): Floats {
  const f = 1 / Math.tan(fovY / 2);
  const nf = 1 / (near - far);
  const m = new Float32Array(16);
  m[0] = f / aspect;
  m[5] = f;
  m[10] = (far + near) * nf;
  m[11] = -1;
  m[14] = 2 * far * near * nf;
  return m;
}

export function ortho(
  left: number,
  right: number,
  bottom: number,
  top: number,
  near: number,
  far: number,
): Floats {
  const m = new Float32Array(16);
  m[0] = 2 / (right - left);
  m[5] = 2 / (top - bottom);
  m[10] = -2 / (far - near);
  m[12] = -(right + left) / (right - left);
  m[13] = -(top + bottom) / (top - bottom);
  m[14] = -(far + near) / (far - near);
  m[15] = 1;
  return m;
}

export function lookAt(eye: Vec3, target: Vec3, up: Vec3): Floats {
  const z = normalize(sub(eye, target));
  const x = normalize(cross(up, z));
  const y = cross(z, x);
  const m = new Float32Array(16);
  m[0] = x[0];
  m[1] = y[0];
  m[2] = z[0];
  m[4] = x[1];
  m[5] = y[1];
  m[6] = z[1];
  m[8] = x[2];
  m[9] = y[2];
  m[10] = z[2];
  m[12] = -dot(x, eye);
  m[13] = -dot(y, eye);
  m[14] = -dot(z, eye);
  m[15] = 1;
  return m;
}

export function multiply(a: Floats, b: Floats): Floats {
  const m = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      m[c * 4 + r] =
        a[r] * b[c * 4] +
        a[r + 4] * b[c * 4 + 1] +
        a[r + 8] * b[c * 4 + 2] +
        a[r + 12] * b[c * 4 + 3];
    }
  }
  return m;
}

export function invert(a: Floats): Floats {
  const [a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, a33] = a;
  const b00 = a00 * a11 - a01 * a10;
  const b01 = a00 * a12 - a02 * a10;
  const b02 = a00 * a13 - a03 * a10;
  const b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11;
  const b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30;
  const b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30;
  const b09 = a21 * a32 - a22 * a31;
  const b10 = a21 * a33 - a23 * a31;
  const b11 = a22 * a33 - a23 * a32;
  const det = 1 / (b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06);
  const m = new Float32Array(16);
  m[0] = (a11 * b11 - a12 * b10 + a13 * b09) * det;
  m[1] = (a02 * b10 - a01 * b11 - a03 * b09) * det;
  m[2] = (a31 * b05 - a32 * b04 + a33 * b03) * det;
  m[3] = (a22 * b04 - a21 * b05 - a23 * b03) * det;
  m[4] = (a12 * b08 - a10 * b11 - a13 * b07) * det;
  m[5] = (a00 * b11 - a02 * b08 + a03 * b07) * det;
  m[6] = (a32 * b02 - a30 * b05 - a33 * b01) * det;
  m[7] = (a20 * b05 - a22 * b02 + a23 * b01) * det;
  m[8] = (a10 * b10 - a11 * b08 + a13 * b06) * det;
  m[9] = (a01 * b08 - a00 * b10 - a03 * b06) * det;
  m[10] = (a30 * b04 - a31 * b02 + a33 * b00) * det;
  m[11] = (a21 * b02 - a20 * b04 - a23 * b00) * det;
  m[12] = (a11 * b07 - a10 * b09 - a12 * b06) * det;
  m[13] = (a00 * b09 - a01 * b07 + a02 * b06) * det;
  m[14] = (a31 * b01 - a30 * b03 - a32 * b00) * det;
  m[15] = (a20 * b03 - a21 * b01 + a22 * b00) * det;
  return m;
}

/** Multiplies a point by a matrix, including the perspective divide. */
export function transformPoint(m: Floats, p: Vec3): Vec3 {
  const w = m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15];
  return [
    (m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12]) / w,
    (m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13]) / w,
    (m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14]) / w,
  ];
}

/** Slab test. Returns the entry and exit distances, or null if the ray misses. */
export function rayBox(
  origin: Vec3,
  dir: Vec3,
  min: Vec3,
  max: Vec3,
): { near: number; far: number } | null {
  let near = -Infinity;
  let far = Infinity;
  for (let i = 0; i < 3; i++) {
    const a = (min[i] - origin[i]) / dir[i];
    const b = (max[i] - origin[i]) / dir[i];
    near = Math.max(near, Math.min(a, b));
    far = Math.min(far, Math.max(a, b));
  }
  return far >= Math.max(near, 0) ? { near, far } : null;
}
