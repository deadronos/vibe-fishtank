import { describe, expect, it } from "vitest";
import { shortRendererName } from "./webgl2";

describe("shortRendererName", () => {
  it.each([
    ["ANGLE (Apple, ANGLE Metal Renderer: Apple M4, Unspecified Version)", "Apple M4"],
    [
      "ANGLE (NVIDIA, NVIDIA GeForce RTX 3080 Direct3D11 vs_5_0 ps_5_0, D3D11)",
      "NVIDIA GeForce RTX 3080",
    ],
    [
      "ANGLE (Intel, Intel(R) UHD Graphics 620 (0x00005917) Direct3D11 vs_5_0 ps_5_0, D3D11)",
      "Intel(R) UHD Graphics 620 (0x00005917)",
    ],
    ["Mali-G78", "Mali-G78"],
    ["ANGLE (unparsable)", "ANGLE (unparsable)"],
  ])("shortens %j", (input, expected) => {
    expect(shortRendererName(input)).toBe(expected);
  });
});
