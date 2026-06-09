# Contributing

## Setup

```sh
pnpm install
pnpm test
pnpm build
```

## Expectations

- Keep the core package framework-neutral.
- Do not add Vue, React, or UI framework dependencies to the published package.
- Add tests for asset resolution, runtime behavior, or viewer state changes when modifying those areas.
- Run `pnpm lint`, `pnpm typecheck`, `pnpm test`, and `pnpm build` before opening a pull request.
