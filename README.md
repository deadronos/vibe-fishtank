# Vibe Fishtank

A planted aquarium rendered in raw WebGL2. There's no 3D engine: the small graphics library in
`src/lib` and everything built on it are written by hand, and every model and texture is generated
in code.

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

- `src/lib`: math, seeded noise, WebGL2 wrappers (programs, instanced meshes, render targets) and a
  procedural mesh builder.
- `src/sim/school.ts`: per-species boids, with walls, obstacles, food and startles.
- `src/scene`: the aquascape and its materials.
  - `common.ts` holds the shared lighting: Beer-Lambert absorption and in-scatter along each view
    ray, plus caustics and shadow lookups.
  - `fish.ts` lofts bodies and fins from a parameter table, swims them in the vertex shader, and
    paints each species' pattern in the fragment shader.
  - `post.ts` resolves 4x MSAA, ray-marches light shafts at half resolution, then applies bloom and
    ACES tone mapping.

Each frame renders the caustic texture, then the shadow map, then the HDR scene, then post. If the
frame rate stays low, the render resolution steps down automatically.
