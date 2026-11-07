import { defineConfig, mergeConfig } from "vitest/config";
import viteConfig from "./vite.config.ts";

export default mergeConfig(
  viteConfig,
  defineConfig({
    test: {
      include: ["src/**/*.test.ts"],
      environment: "node",
      restoreMocks: true,
      coverage: {
        provider: "v8",
        include: ["src/**/*.ts"],
        // These need a real GPU, a canvas or the page, so they're checked in the browser instead:
        // the two device backends, the scene materials that build pipelines, and the app wiring.
        exclude: [
          "src/**/*.test.ts",
          "src/test/**",
          "src/main.ts",
          "src/lib/gpu/webgl2.ts",
          "src/lib/gpu/webgpu.ts",
          "src/lib/gpu/types.ts",
          "src/scene/!(layout).ts",
        ],
        reporter: ["text", "html", "lcov"],
        thresholds: { lines: 90, functions: 90, branches: 85, statements: 90 },
      },
    },
  }),
);
