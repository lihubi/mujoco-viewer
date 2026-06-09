import {
  MujocoRuntimeFacade,
  MujocoThreeViewer,
  buildMujocoBundleFromFiles,
  type MujocoBundle,
  type MujocoFileEntry,
  type MujocoRuntimeFacadeState,
  type MujocoRuntimeHandle,
} from '@likang233/mujoco-viewer'
import './styles.css'

const inlineBundle: MujocoBundle = {
  mjcf: `
<mujoco model="minimal">
  <option timestep="0.01"/>
  <worldbody>
    <light name="key" pos="2 -3 4" dir="-0.4 0.6 -1"/>
    <body name="box" pos="0 0 0.25">
      <joint name="hinge" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
      <geom type="box" size="0.18 0.12 0.08" rgba="0.25 0.62 1 1"/>
    </body>
  </worldbody>
  <actuator>
    <position name="hinge_act" joint="hinge" kp="25"/>
  </actuator>
</mujoco>`,
}

const viewerHost = document.querySelector<HTMLElement>('#viewer')
const statusElement = document.querySelector<HTMLElement>('#status')
const controlsElement = document.querySelector<HTMLElement>('#controls')
const optionsElement = document.querySelector<HTMLElement>('#options')

if (!viewerHost || !statusElement || !controlsElement || !optionsElement) {
  throw new Error('Example DOM is incomplete.')
}

const statusEl = statusElement
const controlsEl = controlsElement
const optionsEl = optionsElement

const viewer = new MujocoThreeViewer({ autoRun: false })
const facade = new MujocoRuntimeFacade()
let runtime: MujocoRuntimeHandle | null = null

viewer.mount(viewerHost)
facade.subscribe(renderState)

const setStatus = (text: string): void => {
  statusEl.textContent = text
}

async function loadBundle(bundle: MujocoBundle): Promise<void> {
  setStatus('Loading...')
  runtime = await viewer.loadBundle(bundle)
  facade.attachRuntime(runtime)
  setStatus('Loaded.')
}

function renderState(state: MujocoRuntimeFacadeState): void {
  controlsEl.replaceChildren(...state.jointItems.concat(state.actuatorItems).map((item) => {
    const wrapper = document.createElement('label')
    wrapper.className = 'slider'
    const title = document.createElement('span')
    title.textContent = `${item.label} ${item.displayValue.toFixed(3)} ${item.unitLabel}`
    const input = document.createElement('input')
    input.type = 'range'
    input.min = String(item.min)
    input.max = String(item.max)
    input.step = '0.001'
    input.value = String(item.displayValue)
    input.disabled = item.disabled
    input.addEventListener('input', () => {
      facade.setControlDisplayValue(item.id, Number(input.value))
    })
    wrapper.append(title, input)
    return wrapper
  }))

  optionsEl.replaceChildren(...state.viewerOptionItems.map((item) => {
    const label = document.createElement('label')
    label.className = 'check'
    const input = document.createElement('input')
    input.type = 'checkbox'
    input.checked = item.enabled
    input.addEventListener('change', () => {
      viewer.setVisualizerOption(item.id, input.checked)
    })
    label.append(input, document.createTextNode(item.label))
    return label
  }))
}

document.querySelector('#load-inline')?.addEventListener('click', () => {
  void loadBundle(inlineBundle).catch((error) => setStatus(String(error)))
})

document.querySelector('#run')?.addEventListener('click', () => runtime?.setRunState('running'))
document.querySelector('#pause')?.addEventListener('click', () => runtime?.setRunState('paused'))
document.querySelector('#step')?.addEventListener('click', () => viewer.stepOnce())
document.querySelector('#reset')?.addEventListener('click', () => viewer.resetScene())

document.querySelector<HTMLInputElement>('#folder-input')?.addEventListener('change', async (event) => {
  const input = event.currentTarget as HTMLInputElement
  const entries: MujocoFileEntry[] = Array.from(input.files ?? []).map((file) => ({
    path: file.webkitRelativePath || file.name,
    file,
  }))
  try {
    await loadBundle(await buildMujocoBundleFromFiles(entries))
  } catch (error) {
    setStatus(error instanceof Error ? error.message : String(error))
  } finally {
    input.value = ''
  }
})

void loadBundle(inlineBundle).catch((error) => setStatus(String(error)))
