# React Integration

The package is framework-neutral. A React component only needs to own the DOM host and dispose the viewer during cleanup.

```tsx
import { useEffect, useRef, useState } from 'react'
import {
  MujocoRuntimeFacade,
  MujocoThreeViewer,
  type MujocoBundle,
  type MujocoRuntimeFacadeState,
} from '@mujoco-web/mujoco-viewer'

export function MujocoReactViewer({ bundle }: { bundle: MujocoBundle }) {
  const hostRef = useRef<HTMLDivElement>(null)
  const [state, setState] = useState<MujocoRuntimeFacadeState | null>(null)

  useEffect(() => {
    if (!hostRef.current) return

    const viewer = new MujocoThreeViewer({ autoRun: true })
    const facade = new MujocoRuntimeFacade()
    const unsubscribe = facade.subscribe(setState)

    viewer.mount(hostRef.current)
    void viewer.loadBundle(bundle).then((runtime) => {
      facade.attachRuntime(runtime)
    })

    return () => {
      unsubscribe()
      facade.dispose()
      viewer.dispose()
    }
  }, [bundle])

  return (
    <>
      <div ref={hostRef} style={{ width: '100%', height: 480 }} />
      <pre>{JSON.stringify(state, null, 2)}</pre>
    </>
  )
}
```

See `examples/react-vite` for a fuller example with controls.
