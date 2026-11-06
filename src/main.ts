/** Wires the simulation, the frame pipeline, and the page's controls together. */

import { OrbitCamera } from "./camera";
import { clamp, lerp3, lookAt, multiply, ortho, rayBox, type Vec3 } from "./lib/math";
import { Caustics } from "./scene/caustics";
import { FrameData, LIGHT_DIR, TANK, WATER_MAX, WATER_MIN, type Drawable } from "./scene/common";
import { Decor } from "./scene/decor";
import { FishRenderer } from "./scene/fish";
import { Particles } from "./scene/particles";
import { Plants } from "./scene/plants";
import { PostPipeline, ShadowMap } from "./scene/post";
import { Room } from "./scene/room";
import { Sand } from "./scene/sand";
import { Glass, WaterSurface } from "./scene/water";
import { School, SPECIES } from "./sim/school";

const LIGHT_DAY: Vec3 = [1.75, 1.68, 1.5];
const LIGHT_NIGHT: Vec3 = [0.13, 0.25, 0.56];
const SCATTER_DAY: Vec3 = [0.022, 0.105, 0.135];
const SCATTER_NIGHT: Vec3 = [0.004, 0.018, 0.05];

function element<T extends HTMLElement>(id: string): T {
  const found = document.getElementById(id);
  if (!found) throw new Error(`Missing #${id}`);
  return found as T;
}

function showError(message: string): void {
  const error = element("error");
  error.textContent = message;
  error.hidden = false;
}

function start(): void {
  const canvas = element<HTMLCanvasElement>("tank");
  const gl = canvas.getContext("webgl2", {
    antialias: false,
    alpha: false,
    depth: false,
    powerPreference: "high-performance",
  });
  if (!gl) {
    showError("This aquarium needs WebGL2, which this browser or device doesn't have available.");
    return;
  }
  canvas.addEventListener("webglcontextlost", (event) => {
    event.preventDefault();
    showError("The graphics context was lost. Reload the page to bring the tank back.");
  });

  gl.disable(gl.CULL_FACE);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

  const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const camera = new OrbitCamera(reducedMotion);
  const frame = new FrameData(gl);
  const caustics = new Caustics(gl);
  const shadowMap = new ShadowMap(gl);
  const post = new PostPipeline(gl);
  const school = new School();

  const room = new Room(gl);
  const sand = new Sand(gl);
  const decor = new Decor(gl);
  const plants = new Plants(gl);
  const fish = new FishRenderer(gl);
  const surface = new WaterSurface(gl);
  const particles = new Particles(gl);
  const glass = new Glass(gl);

  const casters: readonly Drawable[] = [decor, plants, fish];
  const opaque: readonly Drawable[] = [room, sand, decor, plants, fish];

  const lightEye: Vec3 = [LIGHT_DIR[0] * 3, 0.64 + LIGHT_DIR[1] * 3, LIGHT_DIR[2] * 3];
  const lightVP = multiply(
    ortho(-2.1, 2.1, -1.35, 1.35, 0.1, 6),
    lookAt(lightEye, [0, 0.64, 0], [0, 0, -1]),
  );

  element("census").textContent = SPECIES.map((s) => `${s.count} ${s.name}`).join(" · ");

  // Rendering at reduced resolution is the first thing to give on slower machines.
  let resolutionScale = 1;
  let slowFrames = 0;
  const resize = (): void => {
    const dpr = Math.min(window.devicePixelRatio || 1, 2) * resolutionScale;
    const width = Math.max(2, Math.round(canvas.clientWidth * dpr));
    const height = Math.max(2, Math.round(canvas.clientHeight * dpr));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    post.resize(width, height);
  };
  new ResizeObserver(resize).observe(canvas);
  resize();

  let nightTarget = 0;
  let night = 0;
  let time = 0;
  let last = performance.now();

  const render = (now: number): void => {
    const frameMs = now - last;
    last = now;
    const dt = clamp(frameMs / 1000, 0.001, 0.05);
    time += dt;

    if (frameMs > 34 && frameMs < 250) slowFrames += 1;
    else slowFrames = Math.max(0, slowFrames - 0.5);
    if (slowFrames > 75 && resolutionScale > 0.55) {
      resolutionScale *= 0.8;
      slowFrames = 0;
      resize();
    }

    night += (nightTarget - night) * (1 - Math.exp(-dt * 1.4));
    school.update(dt);
    camera.update(dt, canvas.clientWidth / Math.max(canvas.clientHeight, 1));
    fish.update(school.fish);
    particles.update(dt, school.food, camera.right, camera.up);

    frame.set(
      camera.viewProj,
      lightVP,
      camera.position,
      time,
      lerp3(LIGHT_DAY, LIGHT_NIGHT, night),
      1 - night,
      lerp3(SCATTER_DAY, SCATTER_NIGHT, night),
    );

    caustics.render(time);

    frame.setViewProj(lightVP);
    shadowMap.begin();
    for (const item of casters) item.draw(true);

    frame.setViewProj(camera.viewProj);
    post.beginScene();
    gl.enable(gl.DEPTH_TEST);
    gl.depthMask(true);
    for (const item of opaque) item.draw(false);

    gl.enable(gl.BLEND);
    gl.depthMask(false);
    surface.draw(false);
    particles.draw(false);
    glass.draw(false);
    gl.depthMask(true);
    gl.disable(gl.BLEND);

    post.finish(camera.invViewProj, time);
    requestAnimationFrame(render);
  };

  bindControls(canvas, camera, school, (isNight) => {
    nightTarget = isNight ? 1 : 0;
  });
  requestAnimationFrame(render);
}

function bindControls(
  canvas: HTMLCanvasElement,
  camera: OrbitCamera,
  school: School,
  setNight: (night: boolean) => void,
): void {
  const hint = element("hint");
  if (matchMedia("(hover: none)").matches) {
    hint.textContent = "Drag to look around · Pinch to zoom · Tap the water to drop food";
  }
  const dismissHint = (): void => hint.classList.add("gone");
  window.setTimeout(dismissHint, 9000);

  const pointers = new Map<number, { x: number; y: number }>();
  let travelled = 0;
  let pinch = 0;

  canvas.addEventListener("pointerdown", (event) => {
    canvas.setPointerCapture(event.pointerId);
    pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
    travelled = 0;
    pinch = 0;
    canvas.classList.add("dragging");
  });

  canvas.addEventListener("pointermove", (event) => {
    const previous = pointers.get(event.pointerId);
    if (!previous) return;
    const dx = event.clientX - previous.x;
    const dy = event.clientY - previous.y;
    previous.x = event.clientX;
    previous.y = event.clientY;
    travelled += Math.hypot(dx, dy);
    if (pointers.size === 1) {
      camera.orbit(dx, dy);
    } else if (pointers.size === 2) {
      const [a, b] = [...pointers.values()];
      const spread = Math.hypot(a.x - b.x, a.y - b.y);
      if (pinch > 0 && spread > 0) camera.dolly(pinch / spread);
      pinch = spread;
    }
    if (travelled > 8) dismissHint();
  });

  const release = (event: PointerEvent): void => {
    const wasSingle = pointers.size === 1;
    pointers.delete(event.pointerId);
    if (pointers.size === 0) canvas.classList.remove("dragging");
    if (event.type === "pointerup" && wasSingle && travelled < 6) {
      const rect = canvas.getBoundingClientRect();
      const ndcX = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      const ndcY = 1 - ((event.clientY - rect.top) / rect.height) * 2;
      const dir = camera.ray(ndcX, ndcY);
      const hit = rayBox(camera.position, dir, WATER_MIN, WATER_MAX);
      if (hit) {
        const t = Math.max(hit.near, 0);
        school.dropFood(camera.position[0] + dir[0] * t, camera.position[2] + dir[2] * t);
        dismissHint();
      }
    }
  };
  canvas.addEventListener("pointerup", release);
  canvas.addEventListener("pointercancel", release);

  canvas.addEventListener(
    "wheel",
    (event) => {
      event.preventDefault();
      camera.dolly(Math.exp(event.deltaY * 0.0011));
    },
    { passive: false },
  );

  const feed = (): void => {
    school.dropFood(
      (Math.random() - 0.5) * TANK.halfX * 1.3,
      (Math.random() - 0.5) * TANK.halfZ * 1.1,
      9,
    );
    dismissHint();
  };
  const tap = (): void => {
    school.startle([(Math.random() - 0.5) * 1.6, 0.7, TANK.halfZ]);
    dismissHint();
  };
  const lightButton = element<HTMLButtonElement>("light");
  const toggleLight = (): void => {
    const next = lightButton.getAttribute("aria-pressed") !== "true";
    lightButton.setAttribute("aria-pressed", String(next));
    setNight(next);
  };

  element("feed").addEventListener("click", feed);
  element("tap").addEventListener("click", tap);
  lightButton.addEventListener("click", toggleLight);

  window.addEventListener("keydown", (event) => {
    if (event.repeat || event.metaKey || event.ctrlKey || event.altKey) return;
    const key = event.key.toLowerCase();
    if (key === "f") feed();
    else if (key === "t") tap();
    else if (key === "l") toggleLight();
  });
}

try {
  start();
} catch (error) {
  console.error(error);
  showError("The aquarium couldn't start on this device's graphics hardware.");
}
