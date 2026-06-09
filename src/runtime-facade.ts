import type {
  MujocoAngleUnit,
  MujocoControlEditPolicy,
  MujocoRunState,
  MujocoRuntimeControlDescriptor,
  MujocoRuntimeHandle,
  MujocoRuntimeSnapshot,
  MujocoViewerOptionDescriptor,
} from './types'

export interface MujocoControlPanelItem {
  id: string
  kind: 'joint' | 'actuator'
  label: string
  summary: string
  sourceLabel: string
  displayValue: number
  rawValue: number
  min: number
  max: number
  unitLabel: string
  disabled: boolean
  disabledReason: string | null
}

export interface MujocoViewerOptionPanelItem {
  id: string
  label: string
  group: MujocoViewerOptionDescriptor['group']
  category: MujocoViewerOptionDescriptor['category']
  description: string
  enabled: boolean
}

export interface MujocoRuntimeFacadeState {
  isLoading: boolean
  statusText: string
  hasRuntime: boolean
  runState: MujocoRunState
  teachModeEnabled: boolean
  angleUnit: MujocoAngleUnit
  timeSeconds: number
  jointItems: MujocoControlPanelItem[]
  actuatorItems: MujocoControlPanelItem[]
  viewerOptionItems: MujocoViewerOptionPanelItem[]
}

const DEFAULT_SNAPSHOT: MujocoRuntimeSnapshot = {
  runState: 'paused',
  teachModeEnabled: false,
  timeSeconds: 0,
  controlValues: {},
  viewerOptionStates: {},
  interaction: {
    hoveredBodyId: null,
    draggedBodyId: null,
    selectedBodyId: null,
    perturbBodyId: null,
    activePerturbMode: null,
    selectPoint: null,
  },
}

const radToDeg = (value: number): number => value * 180 / Math.PI
const degToRad = (value: number): number => value * Math.PI / 180

const isAngleControl = (descriptor: MujocoRuntimeControlDescriptor): boolean =>
  descriptor.unitKind === 'angle'

const toDisplayValue = (
  descriptor: MujocoRuntimeControlDescriptor,
  rawValue: number,
  angleUnit: MujocoAngleUnit,
): number => {
  if (isAngleControl(descriptor) && angleUnit === 'degree') {
    return radToDeg(rawValue)
  }
  return rawValue
}

const toRawValue = (
  descriptor: MujocoRuntimeControlDescriptor,
  displayValue: number,
  angleUnit: MujocoAngleUnit,
): number => {
  if (isAngleControl(descriptor) && angleUnit === 'degree') {
    return degToRad(displayValue)
  }
  return displayValue
}

const resolveUnitLabel = (
  descriptor: MujocoRuntimeControlDescriptor,
  angleUnit: MujocoAngleUnit,
): string => {
  if (descriptor.unitKind === 'angle') {
    return angleUnit === 'degree' ? 'deg' : 'rad'
  }
  if (descriptor.unitKind === 'distance') {
    return 'm'
  }
  return ''
}

const resolveDisabledReason = (
  editPolicy: MujocoControlEditPolicy,
): string | null => {
  if (editPolicy === 'read-only') {
    return '当前项为只读'
  }
  return null
}

const isControlDisabled = (
  editPolicy: MujocoControlEditPolicy,
  runState: MujocoRunState,
): boolean => editPolicy === 'read-only' || (editPolicy === 'paused-only' && runState === 'running')

const buildPanelItems = (
  descriptors: MujocoRuntimeControlDescriptor[],
  snapshot: MujocoRuntimeSnapshot,
  angleUnit: MujocoAngleUnit,
): MujocoControlPanelItem[] =>
  descriptors.map((descriptor) => {
    const rawValue = snapshot.controlValues[descriptor.id] ?? 0
    const disabledReason = resolveDisabledReason(descriptor.editPolicy)
    return {
      id: descriptor.id,
      kind: descriptor.kind,
      label: descriptor.label,
      summary: descriptor.summary,
      sourceLabel: descriptor.sourceLabel,
      displayValue: toDisplayValue(descriptor, rawValue, angleUnit),
      rawValue,
      min: toDisplayValue(descriptor, descriptor.range.min, angleUnit),
      max: toDisplayValue(descriptor, descriptor.range.max, angleUnit),
      unitLabel: resolveUnitLabel(descriptor, angleUnit),
      disabled: isControlDisabled(descriptor.editPolicy, snapshot.runState),
      disabledReason,
    }
  })

const buildViewerOptionItems = (
  descriptors: MujocoViewerOptionDescriptor[],
  snapshot: MujocoRuntimeSnapshot,
): MujocoViewerOptionPanelItem[] =>
  descriptors.map((descriptor) => ({
    id: descriptor.id,
    label: descriptor.label,
    group: descriptor.group,
    category: descriptor.category,
    description: descriptor.description,
    enabled: snapshot.viewerOptionStates[descriptor.id] ?? descriptor.enabledByDefault,
  }))

const createInitialState = (): MujocoRuntimeFacadeState => ({
  isLoading: false,
  statusText: '等待加载 MJCF',
  hasRuntime: false,
  runState: 'paused',
  teachModeEnabled: false,
  angleUnit: 'radian',
  timeSeconds: 0,
  jointItems: [],
  actuatorItems: [],
  viewerOptionItems: [],
})

export class MujocoRuntimeFacade {
  private readonly listeners = new Set<(state: MujocoRuntimeFacadeState) => void>()
  private state: MujocoRuntimeFacadeState = createInitialState()
  private runtime: MujocoRuntimeHandle | null = null
  private detachRuntimeSubscription: (() => void) | null = null
  private controlDescriptors: MujocoRuntimeControlDescriptor[] = []
  private viewerOptionDescriptors: MujocoViewerOptionDescriptor[] = []
  private snapshot: MujocoRuntimeSnapshot = DEFAULT_SNAPSHOT

  subscribe(listener: (state: MujocoRuntimeFacadeState) => void): () => void {
    this.listeners.add(listener)
    listener(this.state)
    return () => {
      this.listeners.delete(listener)
    }
  }

  getState(): MujocoRuntimeFacadeState {
    return this.state
  }

  setLoadingState(statusText: string, isLoading: boolean): void {
    this.patchState({ statusText, isLoading })
  }

  attachRuntime(runtime: MujocoRuntimeHandle): void {
    this.detachRuntime()
    this.runtime = runtime
    this.controlDescriptors = runtime.getControlDescriptors()
    this.viewerOptionDescriptors = runtime.getViewerOptionDescriptors()
    this.snapshot = runtime.getSnapshot()
    this.detachRuntimeSubscription = runtime.subscribe((snapshot) => {
      this.snapshot = snapshot
      this.rebuildState()
    })
    this.rebuildState()
  }

  detachRuntime(): void {
    this.detachRuntimeSubscription?.()
    this.detachRuntimeSubscription = null
    this.runtime = null
    this.controlDescriptors = []
    this.viewerOptionDescriptors = []
    this.snapshot = DEFAULT_SNAPSHOT
    this.rebuildState()
  }

  setAngleUnit(angleUnit: MujocoAngleUnit): void {
    if (this.state.angleUnit === angleUnit) {
      return
    }
    this.patchState({ angleUnit })
    this.rebuildState()
  }

  setRunState(runState: MujocoRunState): void {
    this.runtime?.setRunState(runState)
  }

  stepOnce(): void {
    this.runtime?.stepOnce()
  }

  resetScene(): void {
    this.runtime?.resetScene()
  }

  setTeachModeEnabled(enabled: boolean): boolean {
    return this.runtime?.setTeachModeEnabled(enabled) ?? false
  }

  setControlDisplayValue(controlId: string, displayValue: number): boolean {
    const descriptor = this.controlDescriptors.find((item) => item.id === controlId)
    if (!descriptor || !this.runtime) {
      return false
    }
    return this.runtime.setControlValue(controlId, toRawValue(descriptor, displayValue, this.state.angleUnit))
  }

  setViewerOptionEnabled(optionId: string, enabled: boolean): boolean {
    return this.runtime?.setViewerOptionEnabled(optionId, enabled) ?? false
  }

  dispose(): void {
    this.detachRuntime()
    this.listeners.clear()
  }

  private rebuildState(): void {
    const panelItems = buildPanelItems(this.controlDescriptors, this.snapshot, this.state.angleUnit)
    this.patchState({
      hasRuntime: this.runtime != null,
      runState: this.snapshot.runState,
      teachModeEnabled: this.snapshot.teachModeEnabled,
      timeSeconds: this.snapshot.timeSeconds,
      jointItems: panelItems.filter((item) => item.kind === 'joint'),
      actuatorItems: panelItems.filter((item) => item.kind === 'actuator'),
      viewerOptionItems: buildViewerOptionItems(this.viewerOptionDescriptors, this.snapshot),
    })
  }

  private patchState(patch: Partial<MujocoRuntimeFacadeState>): void {
    this.state = {
      ...this.state,
      ...patch,
    }
    this.listeners.forEach((listener) => {
      listener(this.state)
    })
  }
}
