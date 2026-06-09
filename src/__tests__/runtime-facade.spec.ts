import { describe, expect, it } from 'vitest'
import { MujocoRuntimeFacade } from '../runtime-facade'
import type {
  MujocoRuntimeControlDescriptor,
  MujocoRuntimeHandle,
  MujocoRuntimeSnapshot,
  MujocoViewerOptionDescriptor,
  MujocoViewerOptionId,
} from '../types'

const controlDescriptors: MujocoRuntimeControlDescriptor[] = [
  {
    id: 'joint:hinge_a',
    kind: 'joint',
    label: 'hinge_a',
    summary: '实时驱动当前关节',
    sourceLabel: 'MuJoCo joint',
    unitKind: 'angle',
    editPolicy: 'paused-only',
    range: { min: -Math.PI, max: Math.PI },
    jointType: 'hinge',
  },
]

const viewerOptionDescriptors: MujocoViewerOptionDescriptor[] = [
  {
    id: 'joint',
    label: '关节',
    description: '显示关节',
    group: 'overlay',
    category: 'joints',
    enabledByDefault: false,
  },
]

const createRuntimeStub = () => {
  let listener: ((snapshot: MujocoRuntimeSnapshot) => void) | null = null
  let latestSnapshot: MujocoRuntimeSnapshot = {
    runState: 'paused',
    teachModeEnabled: false,
    timeSeconds: 0,
    controlValues: {
      'joint:hinge_a': Math.PI / 2,
    },
    viewerOptionStates: {
      joint: true,
    },
    interaction: {
      hoveredBodyId: null,
      draggedBodyId: null,
      selectedBodyId: null,
      perturbBodyId: null,
      activePerturbMode: null,
      selectPoint: null,
    },
  }
  const calls: Array<{ controlId: string; value: number }> = []
  const runtime = {
    controlDescriptors,
    viewerOptionDescriptors,
    executorDescriptors: [],
    viewerOptionState: new Map<MujocoViewerOptionId, boolean>([['joint', true]]),
    interactionState: latestSnapshot.interaction,
    getControlDescriptors: () => controlDescriptors,
    getViewerOptionDescriptors: () => viewerOptionDescriptors,
    getExecutorDescriptors: () => [],
    getSnapshot: () => latestSnapshot,
    subscribe: (nextListener: (snapshot: MujocoRuntimeSnapshot) => void) => {
      listener = nextListener
      nextListener(latestSnapshot)
      return () => {
        listener = null
      }
    },
    setRunState: (runState: 'paused' | 'running') => {
      latestSnapshot = { ...latestSnapshot, runState }
      listener?.(latestSnapshot)
    },
    setTeachModeEnabled: (teachModeEnabled: boolean) => {
      latestSnapshot = { ...latestSnapshot, teachModeEnabled }
      listener?.(latestSnapshot)
      return true
    },
    getTeachModeEnabled: () => latestSnapshot.teachModeEnabled,
    resetScene: () => {},
    stepOnce: () => {},
    tick: () => {},
    forward: () => {},
    setControlValue: (controlId: string, value: number) => {
      calls.push({ controlId, value })
      return true
    },
    setViewerOptionEnabled: () => true,
    setInteractionState: () => {},
    getRuntimeTarget: () => ({ mujoco: {}, model: {}, data: {} }),
    dispose: () => {},
  } as unknown as MujocoRuntimeHandle

  return { runtime, calls }
}

describe('MujocoRuntimeFacade', () => {
  it('builds panel state and converts angle controls between radians and degrees', () => {
    const facade = new MujocoRuntimeFacade()
    const { runtime, calls } = createRuntimeStub()

    facade.attachRuntime(runtime)
    expect(facade.getState().jointItems[0].displayValue).toBeCloseTo(Math.PI / 2)

    facade.setAngleUnit('degree')
    expect(facade.getState().jointItems[0].displayValue).toBeCloseTo(90)
    expect(facade.setControlDisplayValue('joint:hinge_a', 180)).toBe(true)
    expect(calls[0]).toEqual({
      controlId: 'joint:hinge_a',
      value: Math.PI,
    })
  })

  it('disables paused-only controls without a running-state reason', () => {
    const facade = new MujocoRuntimeFacade()
    const { runtime } = createRuntimeStub()

    facade.attachRuntime(runtime)
    expect(facade.getState().jointItems[0].disabled).toBe(false)

    facade.setRunState('running')
    expect(facade.getState().runState).toBe('running')
    expect(facade.getState().jointItems[0].disabled).toBe(true)
    expect(facade.getState().jointItems[0].disabledReason).toBeNull()
  })

  it('syncs teach mode state through the runtime', () => {
    const facade = new MujocoRuntimeFacade()
    const { runtime } = createRuntimeStub()

    facade.attachRuntime(runtime)
    expect(facade.getState().teachModeEnabled).toBe(false)

    expect(facade.setTeachModeEnabled(true)).toBe(true)
    expect(facade.getState().teachModeEnabled).toBe(true)
  })
})
