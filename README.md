# @likang233/mujoco-viewer

Framework-agnostic TypeScript MuJoCo/MJCF viewer for browsers. It uses the official `@mujoco/mujoco` WASM runtime and renders the compiled model with Three.js.

This package does not depend on Vue, Element Plus, React, or any UI framework.

## Install

```sh
npm install @likang233/mujoco-viewer
```

`@mujoco/mujoco` and `three` are regular dependencies, so the package works without extra peer dependency setup. The default loader serves the bundled single-threaded `mujoco.wasm` from `dist/assets/mujoco.wasm`.

## Quick Start

```ts
import { MujocoThreeViewer, type MujocoBundle } from '@likang233/mujoco-viewer'
import mujocoWasmUrl from '@likang233/mujoco-viewer/assets/mujoco.wasm?url'

const host = document.querySelector<HTMLDivElement>('#viewer')
if (!host) throw new Error('Missing viewer host')

const bundle: MujocoBundle = {
  mjcf: `
<mujoco model="minimal">
  <option timestep="0.01"/>
  <worldbody>
    <light pos="0 0 2"/>
    <body name="box" pos="0 0 0.2">
      <joint name="hinge" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
      <geom type="box" size="0.1 0.1 0.1" rgba="0.45 0.7 1 1"/>
    </body>
  </worldbody>
  <actuator>
    <position name="hinge_act" joint="hinge" kp="20"/>
  </actuator>
</mujoco>`,
}

const viewer = new MujocoThreeViewer({
  autoRun: true,
  locateFile: (path) => (path.endsWith('mujoco.wasm') ? mujocoWasmUrl : path),
})
viewer.mount(host)

const runtime = await viewer.loadBundle(bundle)
const unsubscribe = runtime.subscribe((snapshot) => {
  console.log(snapshot.timeSeconds)
})

window.addEventListener('beforeunload', () => {
  unsubscribe()
  viewer.dispose()
})
```

`mount()` appends the canvas, starts the render loop, binds pointer/wheel/double-click/context-menu events, and observes host resize. Call `dispose()` to clean up all event listeners, the MuJoCo runtime, and Three.js resources.

For Vite, import the bundled WASM with `?url` and pass it through `locateFile`, as shown above. Other bundlers need the same idea: serve `mujoco.wasm` as a real static asset and return its URL from `locateFile`.

## Loading a Folder

Use `buildMujocoBundleFromFiles()` when users upload a directory containing `.mjcf` / MuJoCo XML and mesh or texture resources.

```ts
import { buildMujocoBundleFromFiles, MujocoThreeViewer, type MujocoFileEntry } from '@likang233/mujoco-viewer'

const entries: MujocoFileEntry[] = Array.from(input.files ?? []).map((file) => ({
  path: file.webkitRelativePath || file.name,
  file,
}))

const bundle = await buildMujocoBundleFromFiles(entries)
const viewer = new MujocoThreeViewer()
viewer.mount(document.querySelector('#viewer')!)
await viewer.loadBundle(bundle)
```

The asset resolver follows `include`, `compiler assetdir`, `meshdir`, `texturedir`, mesh `file`, texture `file`, and six-file cube texture references. It writes referenced resources into MuJoCo's VFS using stable virtual paths.

## Controls

The runtime exposes descriptors for UI controls. Frameworks can render these however they like.

```ts
const runtime = await viewer.loadBundle(bundle)

for (const control of runtime.getControlDescriptors()) {
  console.log(control.id, control.label, control.range)
}

runtime.setRunState('running')
runtime.setControlValue('joint:0', 0.3)
viewer.setVisualizerOption('contact-point', true)
```

For UI state projection, use `MujocoRuntimeFacade`.

```ts
import { MujocoRuntimeFacade } from '@likang233/mujoco-viewer'

const facade = new MujocoRuntimeFacade()
facade.attachRuntime(runtime)
const unsubscribe = facade.subscribe((state) => {
  renderControls(state.jointItems, state.actuatorItems, state.viewerOptionItems)
})
```

## React

React does not need a special adapter:

```tsx
import { useEffect, useRef } from 'react'
import { MujocoThreeViewer, type MujocoBundle } from '@likang233/mujoco-viewer'
import mujocoWasmUrl from '@likang233/mujoco-viewer/assets/mujoco.wasm?url'

export function MujocoCanvas({ bundle }: { bundle: MujocoBundle }) {
  const hostRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!hostRef.current) return
    const viewer = new MujocoThreeViewer({
      autoRun: true,
      locateFile: (path) => (path.endsWith('mujoco.wasm') ? mujocoWasmUrl : path),
    })
    viewer.mount(hostRef.current)
    void viewer.loadBundle(bundle)
    return () => viewer.dispose()
  }, [bundle])

  return <div ref={hostRef} style={{ width: '100%', height: '100%' }} />
}
```

See `examples/react-vite` for a working app.

To run an example locally:

```sh
npm --prefix examples/vanilla-vite install
npm --prefix examples/vanilla-vite run dev
```

## Examples

- `examples/vanilla-vite`: plain TypeScript + Vite app with upload, run/pause/step/reset, sliders, and viewer options.
- `examples/react-vite`: React + Vite app proving the package has no Vue or Element Plus dependency.
- `examples/assets`: small MJCF assets used by the examples.

## API Documentation

- [API](docs/API.md)
- [Assets and VFS](docs/assets-and-vfs.md)
- [React Integration](docs/react.md)
- [Publishing](docs/publishing.md)

## Browser Support

This is a browser package. It requires DOM, WebGL, `ResizeObserver`, `File`/`Blob` for upload helpers, and modern ESM bundler support.

## License

MIT
