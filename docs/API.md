# API

## `MujocoThreeViewer`

Main DOM-facing viewer class.

```ts
const viewer = new MujocoThreeViewer({
  autoRun: false,
  background: '#000',
})

viewer.mount(hostElement)
const runtime = await viewer.loadBundle(bundle)
viewer.setRunState('running')
viewer.stepOnce()
viewer.resetScene()
viewer.dispose()
```

`mount(host)` appends the canvas, starts rendering, binds interaction events, and observes host resize. `dispose()` releases all runtime, event, and WebGL resources.

## Runtime

`createMujocoRuntime(bundle, options)` loads MuJoCo, writes VFS assets, compiles MJCF, and returns `MujocoRuntimeHandle`.

Important methods:

- `getControlDescriptors()`
- `getViewerOptionDescriptors()`
- `getSnapshot()`
- `subscribe(listener)`
- `setRunState('paused' | 'running')`
- `stepOnce()`
- `resetScene()`
- `setControlValue(controlId, value)`
- `setViewerOptionEnabled(optionId, enabled)`
- `dispose()`

## `MujocoRuntimeFacade`

Framework-neutral state adapter for building control panels.

```ts
const facade = new MujocoRuntimeFacade()
facade.attachRuntime(runtime)
facade.subscribe((state) => {
  console.log(state.jointItems, state.actuatorItems, state.viewerOptionItems)
})
```

It formats angle values, run-state dependent disabled state, and viewer option UI items.

## Bundle Types

```ts
interface MujocoBundle {
  mjcf: string
  metadataXmlSources?: string[]
  modelPath?: string
  vfsPaths?: string[]
  meshAssets?: Array<{
    vfsPath: string
    bytes: Uint8Array
  }>
}

interface MujocoFileEntry {
  path: string
  file: File | Blob
  kind?: string
  id?: string
}
```

## Asset Helpers

- `buildMujocoBundleFromFiles(files, options?)`
- `buildSingleMeshPreviewBundle(entry, options?)`
- `collectReferencedMujocoEntries(files, modelEntry)`
- `collectMujocoMetadataXmlSources(files, modelEntry, rootXmlText?)`
- `buildMujocoVfsAssets(files, modelEntry, vfsRoot)`

## Custom MuJoCo Loader

The default loader uses the package-bundled `dist/assets/mujoco.wasm`. In Vite and similar bundlers, import the WASM as a URL and pass it through `locateFile` so the dev server serves the file with the correct response body and MIME type. Override it the same way when using a CDN, custom WASM build, or multi-threaded MuJoCo build.

```ts
import mujocoWasmUrl from '@likang233/mujoco-viewer/assets/mujoco.wasm?url'

const viewer = new MujocoThreeViewer({
  locateFile: (path) => path.endsWith('mujoco.wasm') ? mujocoWasmUrl : path,
})
```
