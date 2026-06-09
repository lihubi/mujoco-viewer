import type * as THREE from 'three'
import type {
  MainModule as MujocoMainModule,
  MjData,
  MjModel,
  MjvCamera,
  MjvOption,
  MjvPerturb,
  MjvScene,
} from '@mujoco/mujoco'

export type MujocoModule = MujocoMainModule
export type MujocoModel = MjModel
export type MujocoData = MjData

export interface MujocoBundle {
  mjcf: string
  metadataXmlSources?: string[]
  modelPath?: string
  vfsPaths?: string[]
  meshAssets?: Array<{
    vfsPath: string
    bytes: Uint8Array
  }>
}

export interface MujocoFileEntry {
  path: string
  file: File | Blob
  kind?: string
  id?: string
}

export interface MujocoFilePanelEntry {
  path: string
  kind?: string
  size?: number
  selected?: boolean
}

export type MujocoLoader = () => Promise<MujocoModule>
export type MujocoLocateFile = (path: string) => string

export interface MujocoViewerOptions {
  mujocoLoader?: MujocoLoader
  locateFile?: MujocoLocateFile
}

export type MujocoRunState = 'paused' | 'running'
export type MujocoAngleUnit = 'radian' | 'degree'
export type MujocoJointType = 'hinge' | 'slide' | 'ball' | 'free' | 'fixed'
export type MujocoControlKind = 'joint' | 'actuator'
export type MujocoControlEditPolicy = 'paused-only' | 'always' | 'read-only'
export type MujocoControlUnitKind = 'angle' | 'distance' | 'control'
export type MujocoActuatorTransmissionType =
  | 'joint'
  | 'jointinparent'
  | 'slidercrank'
  | 'tendon'
  | 'site'
  | 'body'
  | 'unknown'

export type MujocoViewerOptionGroup = 'overlay' | 'appearance' | 'environment'

export type MujocoViewerOptionCategory =
  | 'joints'
  | 'actuators'
  | 'cameras'
  | 'forces'
  | 'contacts'
  | 'inertia'
  | 'selection'
  | 'transforms'
  | 'materials'
  | 'rendering'
  | 'debug'

export type MujocoViewerOptionId =
  | 'joint'
  | 'actuator'
  | 'tendon'
  | 'site'
  | 'camera-frustum'
  | 'unsupported-placeholders'
  | 'skin'
  | 'flex'
  | 'flex-wireframe'
  | 'sensor-marker'
  | 'equality'
  | 'group-filter'
  | 'inertia'
  | 'perturb-force'
  | 'perturb-object'
  | 'contact-point'
  | 'contact-force'
  | 'transparent'
  | 'center-of-mass'
  | 'select-point'
  | 'shadow'
  | 'wireframe'
  | 'reflection'
  | 'additive'
  | 'fog'

export interface MujocoViewerOptionDescriptor {
  id: MujocoViewerOptionId
  label: string
  description: string
  group: MujocoViewerOptionGroup
  category: MujocoViewerOptionCategory
  enabledByDefault: boolean
}

export type MujocoViewerOptionStateSnapshot = Partial<Record<MujocoViewerOptionId, boolean>>

export interface MujocoNumericRange {
  min: number
  max: number
}

export interface MujocoExecutorDescriptor {
  id: string
  name: string
  jointType: MujocoJointType
  jointTypeLabel: string
  summary: string
  rangeLabel: string
  sourceLabel: string
  range?: [number, number]
  jointNumericId?: number
  bodyId?: number
  qposAddr?: number
  qvelAddr?: number
  dofAdr?: number
  actuatorId?: number
  ctrlAdr?: number
  transmissionType?: MujocoActuatorTransmissionType
}

export interface MujocoRuntimeControlDescriptor {
  id: string
  kind: MujocoControlKind
  label: string
  summary: string
  sourceLabel: string
  unitKind: MujocoControlUnitKind
  editPolicy: MujocoControlEditPolicy
  range: MujocoNumericRange
  jointType?: MujocoJointType
  transmissionType?: MujocoActuatorTransmissionType
  targetLabel?: string
}

export interface MujocoRuntimeSnapshot {
  runState: MujocoRunState
  teachModeEnabled: boolean
  timeSeconds: number
  controlValues: Record<string, number>
  viewerOptionStates: MujocoViewerOptionStateSnapshot
  interaction: MujocoInteractionState
}

export interface MujocoInteractionState {
  hoveredBodyId: number | null
  draggedBodyId: number | null
  selectedBodyId: number | null
  perturbBodyId: number | null
  activePerturbMode: 'translate' | 'rotate' | null
  selectPoint: [number, number, number] | null
}

export interface MujocoRuntimeTarget {
  mujoco: MujocoModule
  model: MujocoModel
  data: MujocoData
}

export interface MujocoRuntimeHandle {
  readonly mujoco: MujocoModule
  readonly model: MujocoModel
  readonly data: MujocoData
  readonly controlDescriptors: MujocoRuntimeControlDescriptor[]
  readonly viewerOptionDescriptors: MujocoViewerOptionDescriptor[]
  readonly executorDescriptors: MujocoExecutorDescriptor[]
  readonly viewerOptionState: Map<MujocoViewerOptionId, boolean>
  readonly interactionState: MujocoInteractionState
  getControlDescriptors(): MujocoRuntimeControlDescriptor[]
  getViewerOptionDescriptors(): MujocoViewerOptionDescriptor[]
  getExecutorDescriptors(): MujocoExecutorDescriptor[]
  getSnapshot(): MujocoRuntimeSnapshot
  subscribe(listener: (snapshot: MujocoRuntimeSnapshot) => void): () => void
  setRunState(runState: MujocoRunState): void
  setTeachModeEnabled(enabled: boolean): boolean
  getTeachModeEnabled(): boolean
  resetScene(): void
  stepOnce(): void
  tick(dtSec: number): void
  forward(): void
  setControlValue(controlId: string, value: number): boolean
  setViewerOptionEnabled(optionId: string, enabled: boolean): boolean
  setInteractionState(patch: Partial<MujocoInteractionState>): void
  getRuntimeTarget(): MujocoRuntimeTarget
  dispose(): void
}

export interface MujocoRuntimeCreateOptions {
  mujocoLoader?: MujocoLoader
  locateFile?: MujocoLocateFile
  initialViewerOptionState?: MujocoViewerOptionStateSnapshot
  initialTeachModeEnabled?: boolean
  modelPath?: string
}

export interface MujocoPickedBodyHit {
  bodyId: number
  geomId: number
  point: THREE.Vector3
}

export interface MujocoForceDragSession {
  bodyId: number
  lastClientX: number
  lastClientY: number
  force: THREE.Vector3
  arrow: THREE.ArrowHelper
  perturb: MjvPerturb
  perturbScene: MjvScene
  perturbCamera: MjvCamera
  perturbOption: MjvOption
}

export interface MujocoFrameContext {
  nowSec: number
  dtSec: number
  phase: 'before-simulation-step' | 'after-scene-sync'
}
