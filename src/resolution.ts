/** Steps render resolution down under sustained load, and back up when it recovers.

    Both directions use hysteresis: a step down needs a sustained stretch of
    genuinely slow frames, and recovery needs an even longer stretch of
    comfortable ones, so the scale never oscillates. */

export class ResolutionController {
  private scale = 1;
  private slowFrames = 0;
  private fastFrames = 0;

  /** Records one frame's duration in milliseconds; true when the scale changed. */
  step(frameMs: number): boolean {
    if (frameMs > 34 && frameMs < 250) {
      // This slow is real load, not a tab switch, which reports much worse.
      this.slowFrames += 1;
      this.fastFrames = 0;
    } else {
      this.slowFrames = Math.max(0, this.slowFrames - 0.5);
      if (this.scale < 1 && frameMs <= 34) this.fastFrames += 1;
    }
    if (this.slowFrames > 75 && this.scale > 0.55) {
      this.scale *= 0.8;
      this.slowFrames = 0;
      this.fastFrames = 0;
      return true;
    }
    if (this.fastFrames > 1500 && this.slowFrames === 0 && this.scale < 1) {
      this.scale = Math.min(1, this.scale / 0.8);
      this.fastFrames = 0;
      return true;
    }
    return false;
  }

  /** The scale factor to multiply the device pixel ratio by. */
  get value(): number {
    return this.scale;
  }
}
