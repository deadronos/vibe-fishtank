import { describe, expect, it } from "vitest";
import { ResolutionController } from "./resolution";

/** Feeds the controller frames of the given duration; true if any stepped. */
const feed = (controller: ResolutionController, ms: number, frames: number): boolean => {
  let changed = false;
  for (let i = 0; i < frames; i++) changed = controller.step(ms) || changed;
  return changed;
};

describe("ResolutionController", () => {
  it("holds full resolution at a healthy frame rate", () => {
    const controller = new ResolutionController();
    expect(feed(controller, 16, 5000)).toBe(false);
    expect(controller.value).toBe(1);
  });

  it("ignores hiccups and tab-switch-sized stalls", () => {
    const controller = new ResolutionController();
    feed(controller, 16, 100);
    feed(controller, 300, 5);
    feed(controller, 16, 100);
    expect(controller.value).toBe(1);
  });

  it("steps down under sustained load, stopping at the floor", () => {
    const controller = new ResolutionController();
    feed(controller, 40, 75);
    expect(controller.value).toBe(1); // 75 slow frames is not yet enough
    expect(controller.step(40)).toBe(true); // the 76th tips it
    expect(controller.value).toBeCloseTo(0.8);
    // 0.8 -> 0.64 -> 0.512, and then the floor holds against more load.
    feed(controller, 40, 1000);
    expect(controller.value).toBeCloseTo(0.512);
    expect(feed(controller, 40, 1000)).toBe(false);
    expect(controller.value).toBeCloseTo(0.512);
  });

  it("recovers only after a long comfortable stretch, one step at a time", () => {
    const controller = new ResolutionController();
    feed(controller, 40, 300);
    expect(controller.value).toBeCloseTo(0.512);
    let before = 0.512;
    for (const expected of [0.64, 0.8, 1]) {
      feed(controller, 16, 1500);
      expect(controller.value).toBeCloseTo(before); // 1500 fast frames is not enough
      expect(controller.step(16)).toBe(true); // the 1501st consecutive tips it
      expect(controller.value).toBeCloseTo(expected);
      before = expected;
    }
    // Back at full resolution, comfort changes nothing further.
    expect(feed(controller, 16, 2000)).toBe(false);
    expect(controller.value).toBe(1);
  });

  it("does not recover while any load remains", () => {
    const controller = new ResolutionController();
    feed(controller, 40, 76);
    feed(controller, 16, 1500);
    expect(controller.value).toBeCloseTo(0.8); // not 1501 consecutive yet
    feed(controller, 16, 1);
    expect(controller.value).toBe(1);
    // A single slow frame resets the recovery streak entirely.
    feed(controller, 40, 76);
    feed(controller, 16, 1500);
    controller.step(40);
    feed(controller, 16, 1500);
    expect(controller.value).toBeCloseTo(0.8);
    expect(controller.step(16)).toBe(true);
    expect(controller.value).toBe(1);
  });

  it("counts only genuinely fast frames toward recovery", () => {
    const controller = new ResolutionController();
    feed(controller, 40, 76);
    // Stalls between the thresholds are load-neutral: they neither step down
    // further nor count as comfort.
    feed(controller, 300, 3000);
    expect(controller.value).toBeCloseTo(0.8);
  });
});
