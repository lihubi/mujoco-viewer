# Assets and VFS

MuJoCo compiles MJCF inside the WASM module and reads external meshes/textures from its virtual filesystem. This package provides helpers that map browser `File`/`Blob` objects to VFS paths.

## File Entries

```ts
interface MujocoFileEntry {
  path: string
  file: File | Blob
  kind?: string
  id?: string
}
```

Use `webkitRelativePath` when files come from a directory input:

```ts
const entries = Array.from(input.files ?? []).map((file) => ({
  path: file.webkitRelativePath || file.name,
  file,
}))
```

## Bundle Creation

```ts
const bundle = await buildMujocoBundleFromFiles(entries, {
  modelPath: 'robot/main.xml',
  vfsRoot: '/mujoco-viewer/scene-1',
})
```

If `modelPath` is omitted, the first `.mjcf` or XML file containing `<mujoco>` is used.

## Reference Resolution

The resolver supports:

- `<include file="...">`
- `<compiler assetdir="...">`
- `<compiler meshdir="...">`
- `<compiler texturedir="...">`
- `<mesh file="...">`
- `<texture file="...">`
- cube texture `fileright`, `fileleft`, `fileup`, `filedown`, `filefront`, `fileback`
- `flexcomp file="..."`

Absolute paths, URLs, `data:` URLs, and `package://` references are intentionally not resolved from local uploads.

## Fallback

If XML parsing fails while collecting assets, `buildMujocoVfsAssets()` falls back to scoped resource files under the selected model directory. This keeps broken or partially edited XML useful during live preview.
