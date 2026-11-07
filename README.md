# Vibe Fishtank

A planted aquarium rendered in raw WebGPU, or WebGL2 where WebGPU isn't available. There's no 3D
engine: the small graphics library in `src/lib` and everything built on it are written by hand, and
every model and texture is generated in code.

## Renderers

The page uses WebGPU when the browser has it, and falls back to WebGL2 otherwise. It also falls back
if WebGPU starts but any of its shaders or pipelines fail to validate. The renderer in use and the
GPU are shown under the title; hover over them for the details and the reason for any fallback.

Add `?renderer=webgl2` or `?renderer=webgpu` to the address to pick one. The other one is still
tried if the one you pick can't start.

## Run it

```bash
pnpm install
pnpm dev
```

`pnpm build` typechecks and produces a static site in `dist/`. `pnpm check` runs the typecheck,
oxlint and oxfmt. The only dependencies are dev tools: TypeScript 7, Vite, oxlint and oxfmt.

## Release

Every `v*` tag deploys the site to GitHub Pages. On a clean, up-to-date `main`, run one of:

```bash
pnpm release patch
```

```bash
pnpm release minor
```

```bash
pnpm release major
```

This runs `pnpm check`, bumps `package.json`, commits, creates the matching `v*` tag, and pushes the
commit and tag. The deploy fails if a tag doesn't match the version in `package.json`.

The repository needs this setup once. It turns on Pages, then allows `v*` tags to deploy, which the
`github-pages` environment doesn't permit by default:

```bash
gh api -X POST repos/deadronos/vibe-fishtank/pages -f build_type=workflow
```

```bash
gh api -X POST repos/deadronos/vibe-fishtank/environments/github-pages/deployment-branch-policies -f name='v*' -f type=tag
```

## Controls

| Input                  | Action                       |
| ---------------------- | ---------------------------- |
| Drag                   | Look around                  |
| Scroll or pinch        | Zoom                         |
| Click the water        | Drop food there              |
| `F` / Feed             | Drop food somewhere random   |
| `T` / Tap glass        | Startle the fish             |
| `L` / Moonlight        | Switch to the night lighting |

## How it's put together

- `src/lib/gpu`: a small, WebGPU-shaped rendering interface (meshes, textures, pipelines, passes
  and per-frame uniforms) with two backends.
  - `webgpu.ts` records each frame into one command encoder, packs uniforms into one buffer
    bound at dynamic offsets, and builds its own mip chains.
  - `webgl2.ts` runs the same calls immediately, with cached framebuffers and blit-based MSAA
    resolve.
  - `index.ts` picks between them and falls back on failure.
- `src/lib`: also math, seeded noise and a procedural mesh builder.
- `src/sim/school.ts`: per-species boids, with walls, obstacles, food and startles.
- `src/scene`: the aquascape and its materials. Each material carries its shaders in GLSL and
  WGSL.
  - `common.ts` holds the shared lighting: Beer-Lambert absorption and in-scatter along each view
    ray, plus caustics and shadow lookups. It also absorbs WebGPU's differences: its 0 to 1 clip
    depth and its top-down texture rows.
  - `fish.ts` lofts bodies and fins from a parameter table, swims them in the vertex shader, and
    paints each species' pattern in the fragment shader.
  - `post.ts` resolves 4x MSAA, ray-marches light shafts at half resolution, then applies bloom and
    ACES tone mapping. WebGPU can't resolve depth, so there the shafts read the multisampled depth
    buffer directly.

Each frame renders the caustic texture, then the shadow map, then the HDR scene, then post. If the
frame rate stays low, the render resolution steps down automatically.
