# Publishing

## Local Checks

```sh
npm ci
npm run lint
npm run typecheck
npm run test
npm run build
npm run examples:install
npm run examples:build
npm pack --dry-run
```

## npm Release

1. Update `CHANGELOG.md`.
2. Bump `package.json` version.
3. Commit and tag:
   ```sh
   git commit -am "release: v0.1.0"
   git tag v0.1.0
   git push origin main --tags
   ```
4. Publish:
   ```sh
   npm publish --access public --provenance
   ```

The GitHub publish workflow expects an `NPM_TOKEN` repository secret.

## GitHub

The repository metadata assumes:

- `https://github.com/mujoco-web/mujoco-viewer`
- npm package `@mujoco-web/mujoco-viewer`

If the GitHub organization changes, update `package.json.repository`, `bugs`, and `homepage`.
