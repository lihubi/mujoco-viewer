import { useEffect, useMemo, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import {
  MujocoRuntimeFacade,
  MujocoThreeViewer,
  type MujocoBundle,
  type MujocoRuntimeFacadeState,
  type MujocoRuntimeHandle,
} from '@mujoco-web/mujoco-viewer'
import './styles.css'

const bundle: MujocoBundle = {
  mjcf: `
<mujoco model="react_minimal">
  <option timestep="0.01"/>
  <worldbody>
    <light name="key" pos="2 -3 4" dir="-0.4 0.6 -1"/>
    <body name="box" pos="0 0 0.25">
      <joint name="hinge" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
      <geom type="box" size="0.18 0.12 0.08" rgba="0.9 0.42 0.35 1"/>
    </body>
  </worldbody>
  <actuator>
    <position name="hinge_act" joint="hinge" kp="25"/>
  </actuator>
</mujoco>`,
}

function App() {
  const hostRef = useRef<HTMLDivElement | null>(null)
  const runtimeRef = useRef<MujocoRuntimeHandle | null>(null)
  const [state, setState] = useState<MujocoRuntimeFacadeState | null>(null)
  const facade = useMemo(() => new MujocoRuntimeFacade(), [])

  useEffect(() => {
    if (!hostRef.current) return undefined

    const viewer = new MujocoThreeViewer({ autoRun: false })
    const unsubscribe = facade.subscribe(setState)
    viewer.mount(hostRef.current)
    void viewer.loadBundle(bundle).then((runtime) => {
      runtimeRef.current = runtime
      facade.attachRuntime(runtime)
    })

    return () => {
      unsubscribe()
      facade.dispose()
      viewer.dispose()
      runtimeRef.current = null
    }
  }, [facade])

  return (
    <main className="app">
      <section ref={hostRef} className="viewer" />
      <aside className="panel">
        <h1>React Example</h1>
        <div className="buttons">
          <button onClick={() => runtimeRef.current?.setRunState('running')}>Run</button>
          <button onClick={() => runtimeRef.current?.setRunState('paused')}>Pause</button>
          <button onClick={() => runtimeRef.current?.stepOnce()}>Step</button>
          <button onClick={() => runtimeRef.current?.resetScene()}>Reset</button>
        </div>
        <p>Time: {state?.timeSeconds.toFixed(3) ?? '0.000'}s</p>
        {state?.jointItems.concat(state.actuatorItems).map((item) => (
          <label key={item.id} className="slider">
            <span>{item.label}: {item.displayValue.toFixed(3)} {item.unitLabel}</span>
            <input
              type="range"
              min={item.min}
              max={item.max}
              step={0.001}
              value={item.displayValue}
              disabled={item.disabled}
              onChange={(event) => facade.setControlDisplayValue(item.id, Number(event.currentTarget.value))}
            />
          </label>
        ))}
      </aside>
    </main>
  )
}

const root = createRoot(document.querySelector('#root')!)
root.render(<App />)
