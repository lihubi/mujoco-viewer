import { applyMujocoControlValue, buildMujocoRuntimeDescriptorCatalog, createMujocoRuntimeSnapshot } from './descriptors'
import { loadMujocoModule, releaseMujocoModule } from './mujoco-loader'
import { MujocoTeachController } from './teach'
import type {
  MujocoBundle,
  MujocoData,
  MujocoInteractionState,
  MujocoModel,
  MujocoModule,
  MujocoRuntimeCreateOptions,
  MujocoRuntimeHandle,
  MujocoRuntimeSnapshot,
  MujocoRuntimeTarget,
  MujocoRunState,
  MujocoViewerOptionId,
} from './types'

type MujocoFs = {
  mkdir(path: string): void
  unlink(path: string): void
  writeFile(path: string, data: string | Uint8Array, options?: { encoding?: string }): void
}

export interface MujocoVfsWriteResult {
  modelPath: string
  writtenPaths: string[]
}

type MujocoConstructors = {
  FS: MujocoFs
  MjModel: {
    mj_loadXML(path: string): MujocoModel
  }
  MjData: new (model: MujocoModel) => MujocoData
  mj_resetData?: (model: MujocoModel, data: MujocoData) => void
  mj_forward(model: MujocoModel, data: MujocoData): void
  mj_step(model: MujocoModel, data: MujocoData): void
}

type MutableNumericArray = {
  length: number
  [index: number]: number
}

type ResettableMujocoData = MujocoData & {
  act?: MutableNumericArray
  ctrl?: MutableNumericArray
  qacc?: MutableNumericArray
  qacc_warmstart?: MutableNumericArray
  qfrc_applied?: MutableNumericArray
  qpos?: MutableNumericArray
  qvel?: MutableNumericArray
  time?: number
  xfrc_applied?: MutableNumericArray
}

type ResettableMujocoModel = MujocoModel & {
  qpos0?: MutableNumericArray
}

type MujocoViewerModelMetadata = {
  __mujocoViewerMaterialTexuniformByName?: Record<string, boolean>
  __mujocoViewerMaterialExplicitLightingByName?: Record<string, {
    specular: boolean
    shininess: boolean
  }>
}

const DEFAULT_MODEL_PATH = '/mujoco-viewer/model.xml'
const TEACH_MODE_SNAPSHOT_INTERVAL_SEC = 1 / 12

const getMujocoConstructors = (mujoco: MujocoModule): MujocoConstructors =>
  mujoco as unknown as MujocoConstructors

const parseMujocoBoolean = (value: string | null): boolean | undefined => {
  if (value == null) {
    return undefined
  }
  const normalized = value.trim().toLowerCase()
  if (normalized === 'true' || normalized === '1') {
    return true
  }
  if (normalized === 'false' || normalized === '0') {
    return false
  }
  return undefined
}

export const parseMaterialTexuniformByName = (mjcf: string): Record<string, boolean> => {
  const byName: Record<string, boolean> = {}
  if (typeof DOMParser !== 'undefined') {
    const document = new DOMParser().parseFromString(mjcf, 'application/xml')
    Array.from(document.getElementsByTagName('material')).forEach((material) => {
      const name = material.getAttribute('name')?.trim()
      const texuniform = parseMujocoBoolean(material.getAttribute('texuniform'))
      if (name && texuniform !== undefined) {
        byName[name] = texuniform
      }
    })
    return byName
  }

  const materialPattern = /<material\b[^>]*>/gi
  const attributePattern = /\b([A-Za-z_:][-A-Za-z0-9_:.]*)\s*=\s*(['"])(.*?)\2/g
  for (const match of mjcf.matchAll(materialPattern)) {
    const attributes: Record<string, string> = {}
    const tag = match[0]
    for (const attributeMatch of tag.matchAll(attributePattern)) {
      attributes[attributeMatch[1]] = attributeMatch[3]
    }
    const name = attributes.name?.trim()
    const texuniform = parseMujocoBoolean(attributes.texuniform ?? null)
    if (name && texuniform !== undefined) {
      byName[name] = texuniform
    }
  }
  return byName
}

export const parseMaterialTexuniformByNameFromSources = (sources: string[]): Record<string, boolean> =>
  sources.reduce<Record<string, boolean>>((merged, source) => ({
    ...merged,
    ...parseMaterialTexuniformByName(source),
  }), {})

export const parseMaterialExplicitLightingByName = (mjcf: string): Record<string, {
  specular: boolean
  shininess: boolean
}> => {
  const byName: Record<string, { specular: boolean; shininess: boolean }> = {}
  if (typeof DOMParser !== 'undefined') {
    const document = new DOMParser().parseFromString(mjcf, 'application/xml')
    Array.from(document.getElementsByTagName('material')).forEach((material) => {
      const name = material.getAttribute('name')?.trim()
      if (name) {
        byName[name] = {
          specular: material.hasAttribute('specular'),
          shininess: material.hasAttribute('shininess'),
        }
      }
    })
    return byName
  }

  const materialPattern = /<material\b[^>]*>/gi
  const attributePattern = /\b([A-Za-z_:][-A-Za-z0-9_:.]*)\s*=\s*(['"])(.*?)\2/g
  for (const match of mjcf.matchAll(materialPattern)) {
    const attributes: Record<string, string> = {}
    const tag = match[0]
    for (const attributeMatch of tag.matchAll(attributePattern)) {
      attributes[attributeMatch[1]] = attributeMatch[3]
    }
    const name = attributes.name?.trim()
    if (name) {
      byName[name] = {
        specular: Object.prototype.hasOwnProperty.call(attributes, 'specular'),
        shininess: Object.prototype.hasOwnProperty.call(attributes, 'shininess'),
      }
    }
  }
  return byName
}

export const parseMaterialExplicitLightingByNameFromSources = (
  sources: string[],
): Record<string, { specular: boolean; shininess: boolean }> =>
  sources.reduce<Record<string, { specular: boolean; shininess: boolean }>>((merged, source) => ({
    ...merged,
    ...parseMaterialExplicitLightingByName(source),
  }), {})

const normalizeVfsPath = (path: string): string => {
  const trimmed = path.trim()
  if (trimmed.length === 0) {
    return trimmed
  }
  return trimmed.startsWith('/') ? trimmed : `/mujoco-viewer/${trimmed}`
}

const ensureVfsDirectory = (fs: MujocoFs, directoryPath: string): void => {
  const normalized = normalizeVfsPath(directoryPath)
  const parts = normalized.split('/').filter(Boolean)
  let cursor = ''
  for (const part of parts) {
    cursor += `/${part}`
    try {
      fs.mkdir(cursor)
    } catch {
      // Existing directories and Emscripten errno variants are safe to ignore.
    }
  }
}

const ensureParentDirectory = (fs: MujocoFs, path: string): void => {
  const index = path.lastIndexOf('/')
  if (index > 0) {
    ensureVfsDirectory(fs, path.slice(0, index))
  }
}

const fillNumericArray = (target: MutableNumericArray | undefined, value = 0): void => {
  if (!target) {
    return
  }
  for (let index = 0; index < target.length; index += 1) {
    target[index] = value
  }
}

const copyNumericArray = (
  target: MutableNumericArray | undefined,
  source: MutableNumericArray | undefined,
): boolean => {
  if (!target || !source) {
    return false
  }
  const count = Math.min(target.length, source.length)
  for (let index = 0; index < count; index += 1) {
    target[index] = Number(source[index] ?? 0)
  }
  for (let index = count; index < target.length; index += 1) {
    target[index] = 0
  }
  return true
}

const resetDataFallback = (model: MujocoModel, data: MujocoData): void => {
  const resettableModel = model as ResettableMujocoModel
  const resettableData = data as ResettableMujocoData
  resettableData.time = 0
  if (!copyNumericArray(resettableData.qpos, resettableModel.qpos0)) {
    fillNumericArray(resettableData.qpos)
  }
  fillNumericArray(resettableData.qvel)
  fillNumericArray(resettableData.qacc)
  fillNumericArray(resettableData.qacc_warmstart)
  fillNumericArray(resettableData.ctrl)
  fillNumericArray(resettableData.act)
  fillNumericArray(resettableData.qfrc_applied)
  fillNumericArray(resettableData.xfrc_applied)
}

export const writeBundleToVfs = (
  mujoco: MujocoModule,
  bundle: MujocoBundle,
  modelPath: string,
): MujocoVfsWriteResult => {
  const { FS } = getMujocoConstructors(mujoco)
  const writtenPaths: string[] = []
  const seenPaths = new Set<string>()
  const rememberPath = (path: string): void => {
    if (seenPaths.has(path)) {
      return
    }
    seenPaths.add(path)
    writtenPaths.push(path)
  }
  const normalizedModelPath = normalizeVfsPath(modelPath)
  ensureParentDirectory(FS, normalizedModelPath)
  FS.writeFile(normalizedModelPath, bundle.mjcf, { encoding: 'utf8' })
  rememberPath(normalizedModelPath)

  bundle.meshAssets?.forEach((asset) => {
    const assetPath = normalizeVfsPath(asset.vfsPath)
    ensureParentDirectory(FS, assetPath)
    FS.writeFile(assetPath, asset.bytes)
    rememberPath(assetPath)
  })

  bundle.vfsPaths?.forEach((path) => {
    rememberPath(normalizeVfsPath(path))
  })

  return {
    modelPath: normalizedModelPath,
    writtenPaths,
  }
}

export const cleanupVfsPaths = (mujoco: MujocoModule, paths: string[]): void => {
  const { FS } = getMujocoConstructors(mujoco)
  const normalizedPaths = [...new Set(paths.map((path) => normalizeVfsPath(path)))]
  normalizedPaths.reverse().forEach((path) => {
    try {
      FS.unlink(path)
    } catch {
      // Missing or already-deleted VFS files must not block runtime disposal.
    }
  })
}

class MujocoRuntime implements MujocoRuntimeHandle {
  readonly viewerOptionState: Map<MujocoViewerOptionId, boolean>
  readonly interactionState: MujocoInteractionState = {
    hoveredBodyId: null,
    draggedBodyId: null,
    selectedBodyId: null,
    perturbBodyId: null,
    activePerturbMode: null,
    selectPoint: null,
  }

  private readonly snapshotListeners = new Set<(snapshot: MujocoRuntimeSnapshot) => void>()
  private readonly teachController: MujocoTeachController
  private runStateValue: MujocoRunState = 'paused'
  private simulationCarryoverSec = 0
  private snapshotCarryoverSec = 0
  private isDisposed = false

  constructor(
    readonly mujoco: MujocoModule,
    readonly model: MujocoModel,
    readonly data: MujocoData,
    readonly controlDescriptors: MujocoRuntimeHandle['controlDescriptors'],
    readonly viewerOptionDescriptors: MujocoRuntimeHandle['viewerOptionDescriptors'],
    readonly executorDescriptors: MujocoRuntimeHandle['executorDescriptors'],
    private readonly vfsPaths: string[],
    private readonly releaseModuleOnDispose: boolean,
    initialViewerOptionState?: MujocoRuntimeCreateOptions['initialViewerOptionState'],
    initialTeachModeEnabled?: MujocoRuntimeCreateOptions['initialTeachModeEnabled'],
  ) {
    this.teachController = new MujocoTeachController({
      target: {
        mujoco: this.mujoco,
        model: this.model,
        data: this.data,
      },
      executorDescriptors,
    })
    this.viewerOptionState = new Map(
      viewerOptionDescriptors.map((descriptor) => [
        descriptor.id,
        descriptor.enabledByDefault,
      ]),
    )
    if (initialViewerOptionState) {
      Object.entries(initialViewerOptionState).forEach(([optionId, enabled]) => {
        if (typeof enabled === 'boolean') {
          this.viewerOptionState.set(optionId as MujocoViewerOptionId, enabled)
        }
      })
    }
    if (initialTeachModeEnabled === true) {
      this.teachController.setEnabled(true)
    }
  }

  getControlDescriptors(): MujocoRuntimeHandle['controlDescriptors'] {
    return this.controlDescriptors
  }

  getViewerOptionDescriptors(): MujocoRuntimeHandle['viewerOptionDescriptors'] {
    return this.viewerOptionDescriptors
  }

  getExecutorDescriptors(): MujocoRuntimeHandle['executorDescriptors'] {
    return this.executorDescriptors
  }

  getSnapshot(): MujocoRuntimeSnapshot {
    return createMujocoRuntimeSnapshot({
      controlDescriptors: this.controlDescriptors,
      executorDescriptors: this.executorDescriptors,
      data: this.data,
      runState: this.runStateValue,
      teachModeEnabled: this.teachController.enabled,
      viewerOptionState: this.viewerOptionState,
      interaction: { ...this.interactionState },
    })
  }

  subscribe(listener: (snapshot: MujocoRuntimeSnapshot) => void): () => void {
    this.snapshotListeners.add(listener)
    listener(this.getSnapshot())
    return () => {
      this.snapshotListeners.delete(listener)
    }
  }

  setRunState(runState: MujocoRunState): void {
    if (this.isDisposed || this.runStateValue === runState) {
      return
    }
    this.runStateValue = runState
    this.simulationCarryoverSec = 0
    this.snapshotCarryoverSec = 0
    this.emitSnapshot()
  }

  setTeachModeEnabled(enabled: boolean): boolean {
    if (this.isDisposed) {
      return false
    }
    const changed = this.teachController.setEnabled(enabled)
    if (changed) {
      this.simulationCarryoverSec = 0
      this.snapshotCarryoverSec = 0
      this.emitSnapshot()
    }
    return changed
  }

  getTeachModeEnabled(): boolean {
    return this.teachController.enabled
  }

  resetScene(): void {
    if (this.isDisposed) {
      return
    }
    const currentRunState = this.runStateValue
    this.simulationCarryoverSec = 0
    this.snapshotCarryoverSec = 0
    Object.assign(this.interactionState, {
      hoveredBodyId: null,
      draggedBodyId: null,
      selectedBodyId: null,
      perturbBodyId: null,
      activePerturbMode: null,
      selectPoint: null,
    })
    const constructors = getMujocoConstructors(this.mujoco)
    if (constructors.mj_resetData) {
      constructors.mj_resetData(this.model, this.data)
    } else {
      resetDataFallback(this.model, this.data)
    }
    this.forward()
    this.runStateValue = currentRunState
    this.emitSnapshot()
  }

  stepOnce(): void {
    if (this.isDisposed || this.runStateValue === 'running') {
      return
    }
    this.beforeSimulationStep()
    getMujocoConstructors(this.mujoco).mj_step(this.model, this.data)
    this.forward()
    this.emitSnapshot()
  }

  tick(dtSec: number): void {
    if (this.isDisposed || this.runStateValue !== 'running') {
      return
    }

    const timestep = Math.max(Number((this.model as { opt?: { timestep?: number } }).opt?.timestep ?? 0.01), 1e-4)
    this.simulationCarryoverSec += Math.max(0, dtSec)
    let iterations = 0
    while (this.simulationCarryoverSec >= timestep && iterations < 8) {
      this.beforeSimulationStep()
      getMujocoConstructors(this.mujoco).mj_step(this.model, this.data)
      this.simulationCarryoverSec -= timestep
      iterations += 1
    }
    this.forward()
    this.emitRunningSnapshot(dtSec)
  }

  forward(): void {
    if (this.isDisposed) {
      return
    }
    getMujocoConstructors(this.mujoco).mj_forward(this.model, this.data)
  }

  setControlValue(controlId: string, value: number): boolean {
    if (this.isDisposed) {
      return false
    }
    const changed = applyMujocoControlValue({
      controlId,
      value,
      runState: this.runStateValue,
      controlDescriptors: this.controlDescriptors,
      executorDescriptors: this.executorDescriptors,
      data: this.data,
    })
    if (!changed) {
      return false
    }
    this.forward()
    this.emitSnapshot()
    return true
  }

  setViewerOptionEnabled(optionId: string, enabled: boolean): boolean {
    if (this.isDisposed) {
      return false
    }
    const descriptor = this.viewerOptionDescriptors.find((item) => item.id === optionId)
    if (!descriptor) {
      return false
    }
    this.viewerOptionState.set(descriptor.id, enabled)
    this.emitSnapshot()
    return true
  }

  setInteractionState(patch: Partial<MujocoInteractionState>): void {
    if (this.isDisposed) {
      return
    }
    Object.assign(this.interactionState, patch)
    this.snapshotCarryoverSec = 0
    this.emitSnapshot()
  }

  getRuntimeTarget(): MujocoRuntimeTarget {
    return {
      mujoco: this.mujoco,
      model: this.model,
      data: this.data,
    }
  }

  dispose(): void {
    if (this.isDisposed) {
      return
    }
    this.snapshotListeners.clear()
    this.isDisposed = true
    this.teachController.dispose()
    try {
      try {
        this.data.delete()
      } finally {
        try {
          this.model.delete()
        } finally {
          cleanupVfsPaths(this.mujoco, this.vfsPaths)
        }
      }
    } finally {
      if (this.releaseModuleOnDispose) {
        releaseMujocoModule(this.mujoco)
      }
    }
  }

  private beforeSimulationStep(): void {
    this.teachController.beforeStep({
      perturbBodyId: this.interactionState.activePerturbMode
        ? null
        : this.interactionState.perturbBodyId,
    })
  }

  private emitSnapshot(): void {
    const snapshot = this.getSnapshot()
    this.snapshotListeners.forEach((listener) => {
      listener(snapshot)
    })
  }

  private emitRunningSnapshot(dtSec: number): void {
    if (!this.teachController.enabled) {
      this.emitSnapshot()
      return
    }
    this.snapshotCarryoverSec += Math.max(0, dtSec)
    if (this.snapshotCarryoverSec < TEACH_MODE_SNAPSHOT_INTERVAL_SEC) {
      return
    }
    this.snapshotCarryoverSec = 0
    this.emitSnapshot()
  }
}

export const createMujocoRuntime = async (
  bundle: MujocoBundle,
  options: MujocoRuntimeCreateOptions = {},
): Promise<MujocoRuntimeHandle> => {
  const mujoco = await loadMujocoModule({
    mujocoLoader: options.mujocoLoader,
    locateFile: options.locateFile,
  })
  const vfsWrite = writeBundleToVfs(mujoco, bundle, options.modelPath ?? bundle.modelPath ?? DEFAULT_MODEL_PATH)
  const constructors = getMujocoConstructors(mujoco)
  try {
    const model = constructors.MjModel.mj_loadXML(vfsWrite.modelPath)
    const metadataXmlSources = bundle.metadataXmlSources ?? [bundle.mjcf]
    ;(model as MujocoViewerModelMetadata).__mujocoViewerMaterialTexuniformByName = parseMaterialTexuniformByNameFromSources(metadataXmlSources)
    ;(model as MujocoViewerModelMetadata).__mujocoViewerMaterialExplicitLightingByName = parseMaterialExplicitLightingByNameFromSources(metadataXmlSources)
    const data = new constructors.MjData(model)
    constructors.mj_forward(model, data)
    const catalog = buildMujocoRuntimeDescriptorCatalog(mujoco, model)
    return new MujocoRuntime(
      mujoco,
      model,
      data,
      catalog.controlDescriptors,
      catalog.viewerOptionDescriptors,
      catalog.executorDescriptors,
      vfsWrite.writtenPaths,
      !options.mujocoLoader,
      options.initialViewerOptionState,
      options.initialTeachModeEnabled,
    )
  } catch (error) {
    cleanupVfsPaths(mujoco, vfsWrite.writtenPaths)
    if (!options.mujocoLoader) {
      releaseMujocoModule(mujoco)
    }
    throw error
  }
}
