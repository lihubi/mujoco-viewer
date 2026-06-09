import {
  DEFAULT_MUJOCO_VIEWER_OPTION_DESCRIPTORS,
} from './viewer-options'
import type {
  MujocoControlUnitKind,
  MujocoActuatorTransmissionType,
  MujocoExecutorDescriptor,
  MujocoJointType,
  MujocoModule,
  MujocoRuntimeControlDescriptor,
  MujocoRuntimeSnapshot,
  MujocoRunState,
  MujocoViewerOptionId,
  MujocoViewerOptionDescriptor,
} from './types'

type NumericReader = {
  length: number
  [index: number]: number
}

type MujocoModelReader = {
  names?: NumericReader
  njnt?: number
  nu?: number
  nactuator?: number
  jnt_type?: NumericReader
  jnt_range?: NumericReader
  jnt_bodyid?: NumericReader
  jnt_qposadr?: NumericReader
  jnt_dofadr?: NumericReader
  name_jntadr?: NumericReader
  ntendon?: number
  nsite?: number
  nbody?: number
  actuator_trnid?: NumericReader
  actuator_trntype?: NumericReader
  actuator_ctrlrange?: NumericReader
  name_actuatoradr?: NumericReader
  name_tendonadr?: NumericReader
  name_siteadr?: NumericReader
  name_bodyadr?: NumericReader
}

type MujocoDataReader = {
  time?: number
  qpos?: NumericReader
  qvel?: NumericReader
  ctrl?: NumericReader
}

export interface MujocoRuntimeDescriptorCatalog {
  controlDescriptors: MujocoRuntimeControlDescriptor[]
  executorDescriptors: MujocoExecutorDescriptor[]
  viewerOptionDescriptors: MujocoViewerOptionDescriptor[]
}

export const clampNumber = (value: number, min: number, max: number): number =>
  Math.min(max, Math.max(min, value))

export const readMujocoName = (model: MujocoModelReader, adr: number): string => {
  if (adr == null || adr < 0 || !model.names) {
    return ''
  }

  let text = ''
  let index = adr
  let safety = 0
  while (model.names[index] !== 0 && safety < 256) {
    text += String.fromCharCode(model.names[index])
    index += 1
    safety += 1
  }
  return text
}

const readEnumValue = (entry: unknown, fallback: number): number => {
  const maybeValue = entry as { value?: unknown } | null
  return typeof maybeValue?.value === 'number' ? maybeValue.value : fallback
}

const getMujocoJointEnumValues = (mujoco: MujocoModule) => {
  const enumSource = (mujoco as unknown as {
    mjtJoint?: Record<string, unknown>
  }).mjtJoint
  return {
    free: readEnumValue(enumSource?.mjJNT_FREE, 0),
    ball: readEnumValue(enumSource?.mjJNT_BALL, 1),
    slide: readEnumValue(enumSource?.mjJNT_SLIDE, 2),
    hinge: readEnumValue(enumSource?.mjJNT_HINGE, 3),
  }
}

const getMujocoTransmissionEnumValues = (mujoco: MujocoModule) => {
  const enumSource = (mujoco as unknown as {
    mjtTrn?: Record<string, unknown>
  }).mjtTrn
  return {
    joint: readEnumValue(enumSource?.mjTRN_JOINT, 0),
    jointinparent: readEnumValue(enumSource?.mjTRN_JOINTINPARENT, 1),
    slidercrank: readEnumValue(enumSource?.mjTRN_SLIDERCRANK, 2),
    tendon: readEnumValue(enumSource?.mjTRN_TENDON, 3),
    site: readEnumValue(enumSource?.mjTRN_SITE, 4),
    body: readEnumValue(enumSource?.mjTRN_BODY, 5),
  }
}

const resolveTransmissionType = (
  mujoco: MujocoModule,
  model: MujocoModelReader,
  actuatorId: number,
): MujocoActuatorTransmissionType => {
  const rawType = Number(model.actuator_trntype?.[actuatorId] ?? getMujocoTransmissionEnumValues(mujoco).joint)
  const enums = getMujocoTransmissionEnumValues(mujoco)
  if (rawType === enums.joint) {
    return 'joint'
  }
  if (rawType === enums.jointinparent) {
    return 'jointinparent'
  }
  if (rawType === enums.slidercrank) {
    return 'slidercrank'
  }
  if (rawType === enums.tendon) {
    return 'tendon'
  }
  if (rawType === enums.site) {
    return 'site'
  }
  if (rawType === enums.body) {
    return 'body'
  }
  return 'unknown'
}

const resolveJointType = (
  mujoco: MujocoModule,
  model: MujocoModelReader,
  jointNumericId: number,
): MujocoJointType => {
  const rawType = Number(model.jnt_type?.[jointNumericId] ?? -1)
  const enums = getMujocoJointEnumValues(mujoco)
  if (rawType === enums.hinge) {
    return 'hinge'
  }
  if (rawType === enums.slide) {
    return 'slide'
  }
  if (rawType === enums.ball) {
    return 'ball'
  }
  if (rawType === enums.free) {
    return 'free'
  }
  return 'fixed'
}

const formatJointTypeLabel = (jointType: MujocoJointType): string => {
  if (jointType === 'hinge') {
    return 'Hinge / 旋转'
  }
  if (jointType === 'slide') {
    return 'Slide / 平移'
  }
  if (jointType === 'ball') {
    return 'Ball / 球关节'
  }
  if (jointType === 'free') {
    return 'Free / 自由关节'
  }
  return 'Fixed / 固定关节'
}

const formatRangeLabel = (range?: [number, number]): string => {
  if (!range) {
    return '未配置关节 limit'
  }
  return `控制范围 ${range[0].toFixed(2)} ~ ${range[1].toFixed(2)}`
}

const normalizeRange = (
  range: [number, number] | undefined,
  fallback: [number, number],
): { min: number; max: number } => {
  const source = range ?? fallback
  const [a, b] = source
  if (!Number.isFinite(a) || !Number.isFinite(b) || Math.abs(a - b) <= 1e-9) {
    return { min: fallback[0], max: fallback[1] }
  }
  return a <= b ? { min: a, max: b } : { min: b, max: a }
}

const readFiniteRange = (
  source: NumericReader | undefined,
  offset: number,
): [number, number] | undefined => {
  const min = Number(source?.[offset])
  const max = Number(source?.[offset + 1])
  if (!Number.isFinite(min) || !Number.isFinite(max) || Math.abs(max - min) <= 1e-9) {
    return undefined
  }
  return min <= max ? [min, max] : [max, min]
}

const resolveUnitKind = (
  jointType: MujocoJointType | undefined,
  transmissionType?: MujocoActuatorTransmissionType,
): MujocoControlUnitKind => {
  if (transmissionType && transmissionType !== 'joint' && transmissionType !== 'jointinparent') {
    return 'control'
  }
  if (jointType === 'hinge') {
    return 'angle'
  }
  if (jointType === 'slide') {
    return 'distance'
  }
  return 'control'
}

const formatTransmissionLabel = (transmissionType: MujocoActuatorTransmissionType): string => {
  if (transmissionType === 'joint') {
    return 'joint'
  }
  if (transmissionType === 'jointinparent') {
    return 'jointinparent'
  }
  if (transmissionType === 'slidercrank') {
    return 'slidercrank'
  }
  if (transmissionType === 'tendon') {
    return 'tendon'
  }
  if (transmissionType === 'site') {
    return 'site'
  }
  if (transmissionType === 'body') {
    return 'body'
  }
  return 'unknown'
}

const resolveTransmissionTargetLabel = (
  model: MujocoModelReader,
  transmissionType: MujocoActuatorTransmissionType,
  targetId: number,
): string => {
  if (targetId < 0) {
    return '未指定 target'
  }
  if (transmissionType === 'joint' || transmissionType === 'jointinparent') {
    return readMujocoName(model, Number(model.name_jntadr?.[targetId] ?? -1)) || `joint_${targetId}`
  }
  if (transmissionType === 'tendon') {
    return readMujocoName(model, Number(model.name_tendonadr?.[targetId] ?? -1)) || `tendon_${targetId}`
  }
  if (transmissionType === 'site' || transmissionType === 'slidercrank') {
    return readMujocoName(model, Number(model.name_siteadr?.[targetId] ?? -1)) || `site_${targetId}`
  }
  if (transmissionType === 'body') {
    return readMujocoName(model, Number(model.name_bodyadr?.[targetId] ?? -1)) || `body_${targetId}`
  }
  return `target_${targetId}`
}

const readActuatorName = (
  model: MujocoModelReader,
  actuatorId: number,
): string => readMujocoName(
  model,
  Number(model.name_actuatoradr?.[actuatorId] ?? -1),
) || `actuator_${actuatorId}`

const resolveActuatorRange = (
  model: MujocoModelReader,
  executor: MujocoExecutorDescriptor,
): { min: number; max: number } => {
  if (executor.actuatorId != null) {
    const range = readFiniteRange(model.actuator_ctrlrange, executor.actuatorId * 2)
    return normalizeRange(range, executor.range ?? [-1, 1])
  }
  return normalizeRange(executor.range, [-1, 1])
}

export const buildRuntimeExecutorsFromModel = (
  mujoco: MujocoModule,
  rawModel: unknown,
): MujocoExecutorDescriptor[] => {
  const model = rawModel as MujocoModelReader
  const actuatorByJointId = new Map<
    number,
    {
      actuatorId: number
      ctrlAdr: number
      actuatorName: string
      transmissionType: MujocoActuatorTransmissionType
    }
  >()

  const actuatorCount = Number(model.nu ?? model.nactuator ?? 0)
  for (let actuatorId = 0; actuatorId < actuatorCount; actuatorId += 1) {
    const transmissionType = resolveTransmissionType(mujoco, model, actuatorId)
    if (transmissionType !== 'joint' && transmissionType !== 'jointinparent') {
      continue
    }
    const jointNumericId = Number(model.actuator_trnid?.[actuatorId * 2] ?? -1)
    if (jointNumericId < 0) {
      continue
    }

    const actuatorName = readActuatorName(model, actuatorId)
    actuatorByJointId.set(jointNumericId, {
      actuatorId,
      ctrlAdr: actuatorId,
      actuatorName,
      transmissionType,
    })
  }

  const executors: MujocoExecutorDescriptor[] = []
  const jointCount = Number(model.njnt ?? 0)
  for (let jointNumericId = 0; jointNumericId < jointCount; jointNumericId += 1) {
    const jointType = resolveJointType(mujoco, model, jointNumericId)
    if (jointType !== 'hinge' && jointType !== 'slide') {
      continue
    }

    const range = readFiniteRange(model.jnt_range, jointNumericId * 2) ?? [-Math.PI, Math.PI]
    const jointName = readMujocoName(
      model,
      Number(model.name_jntadr?.[jointNumericId] ?? -1),
    ) || `joint_${jointNumericId}`
    const actuatorEntry = actuatorByJointId.get(jointNumericId)

    executors.push({
      id: jointName,
      name: jointName,
      jointType,
      jointTypeLabel: formatJointTypeLabel(jointType),
      summary: actuatorEntry?.actuatorName
        ? `实时驱动 ${actuatorEntry.actuatorName}`
        : '实时驱动当前关节',
      rangeLabel: formatRangeLabel(range),
      sourceLabel: actuatorEntry?.actuatorName || 'MuJoCo joint',
      range,
      jointNumericId,
      bodyId: Number(model.jnt_bodyid?.[jointNumericId] ?? -1),
      qposAddr: Number(model.jnt_qposadr?.[jointNumericId] ?? -1),
      qvelAddr: Number(model.jnt_dofadr?.[jointNumericId] ?? -1),
      dofAdr: Number(model.jnt_dofadr?.[jointNumericId] ?? -1),
      actuatorId: actuatorEntry?.actuatorId,
      ctrlAdr: actuatorEntry?.ctrlAdr,
      transmissionType: actuatorEntry?.transmissionType,
    })
  }

  return executors
}

export const buildMujocoRuntimeDescriptorCatalog = (
  mujoco: MujocoModule,
  rawModel: unknown,
): MujocoRuntimeDescriptorCatalog => {
  const model = rawModel as MujocoModelReader
  const executorDescriptors = buildRuntimeExecutorsFromModel(mujoco, model)
  const jointDescriptors: MujocoRuntimeControlDescriptor[] = executorDescriptors.map((executor) => ({
    id: `joint:${executor.id}`,
    kind: 'joint',
    label: executor.name,
    summary: executor.summary,
    sourceLabel: executor.sourceLabel,
    unitKind: resolveUnitKind(executor.jointType),
    editPolicy: 'paused-only',
    range: normalizeRange(executor.range, [-Math.PI, Math.PI]),
    jointType: executor.jointType,
  }))
  const actuatorDescriptors: MujocoRuntimeControlDescriptor[] = executorDescriptors
    .filter((executor) => executor.actuatorId != null && executor.ctrlAdr != null)
    .map((executor) => ({
      id: `actuator:${executor.actuatorId}`,
      kind: 'actuator',
      label: executor.sourceLabel,
      summary: `控制 ${executor.name}`,
      sourceLabel: executor.name,
      unitKind: resolveUnitKind(executor.jointType, executor.transmissionType),
      editPolicy: 'always',
      range: resolveActuatorRange(model, executor),
      jointType: executor.jointType,
      transmissionType: executor.transmissionType ?? 'joint',
      targetLabel: executor.name,
    }))
  const jointActuatorIds = new Set(
    actuatorDescriptors.map((descriptor) => Number(descriptor.id.replace(/^actuator:/, ''))),
  )
  const standaloneActuatorDescriptors: MujocoRuntimeControlDescriptor[] = []
  const actuatorCount = Number(model.nu ?? model.nactuator ?? 0)
  for (let actuatorId = 0; actuatorId < actuatorCount; actuatorId += 1) {
    if (jointActuatorIds.has(actuatorId)) {
      continue
    }
    const transmissionType = resolveTransmissionType(mujoco, model, actuatorId)
    const targetId = Number(model.actuator_trnid?.[actuatorId * 2] ?? -1)
    const actuatorName = readActuatorName(model, actuatorId)
    const targetLabel = resolveTransmissionTargetLabel(model, transmissionType, targetId)
    standaloneActuatorDescriptors.push({
      id: `actuator:${actuatorId}`,
      kind: 'actuator',
      label: actuatorName,
      summary: `控制 ${formatTransmissionLabel(transmissionType)} transmission：${targetLabel}`,
      sourceLabel: `${formatTransmissionLabel(transmissionType)}:${targetLabel}`,
      unitKind: 'control',
      editPolicy: 'always',
      range: normalizeRange(readFiniteRange(model.actuator_ctrlrange, actuatorId * 2), [-1, 1]),
      transmissionType,
      targetLabel,
    })
  }

  return {
    controlDescriptors: [...jointDescriptors, ...actuatorDescriptors, ...standaloneActuatorDescriptors],
    executorDescriptors,
    viewerOptionDescriptors: DEFAULT_MUJOCO_VIEWER_OPTION_DESCRIPTORS,
  }
}

export const createMujocoRuntimeSnapshot = (options: {
  controlDescriptors: MujocoRuntimeControlDescriptor[]
  executorDescriptors: MujocoExecutorDescriptor[]
  data: MujocoDataReader
  runState: MujocoRunState
  teachModeEnabled: boolean
  viewerOptionState: Map<MujocoViewerOptionId, boolean>
  interaction: MujocoRuntimeSnapshot['interaction']
}): MujocoRuntimeSnapshot => {
  const controlValueById: Record<string, number> = {}
  const executorByJointId = new Map(options.executorDescriptors.map((item) => [item.id, item]))

  options.controlDescriptors.forEach((descriptor) => {
    if (descriptor.kind === 'joint') {
      const jointId = descriptor.id.replace(/^joint:/, '')
      const executor = executorByJointId.get(jointId)
      const qposAdr = executor?.qposAddr ?? -1
      controlValueById[descriptor.id] = qposAdr >= 0
        ? Number(options.data.qpos?.[qposAdr] ?? 0)
        : 0
      return
    }

    const actuatorId = Number(descriptor.id.replace(/^actuator:/, ''))
    controlValueById[descriptor.id] = Number(options.data.ctrl?.[actuatorId] ?? 0)
  })

  return {
    runState: options.runState,
    teachModeEnabled: options.teachModeEnabled,
    timeSeconds: Number(options.data.time ?? 0),
    controlValues: controlValueById,
    viewerOptionStates: Object.fromEntries(options.viewerOptionState.entries()),
    interaction: options.interaction,
  }
}

export const applyMujocoControlValue = (options: {
  controlId: string
  value: number
  runState: MujocoRunState
  controlDescriptors: MujocoRuntimeControlDescriptor[]
  executorDescriptors: MujocoExecutorDescriptor[]
  data: MujocoDataReader
}): boolean => {
  const descriptor = options.controlDescriptors.find((item) => item.id === options.controlId)
  if (!descriptor) {
    return false
  }

  if (descriptor.editPolicy === 'read-only') {
    return false
  }
  if (descriptor.editPolicy === 'paused-only' && options.runState === 'running') {
    return false
  }

  const clampedValue = clampNumber(
    Number.isFinite(options.value) ? options.value : 0,
    descriptor.range.min,
    descriptor.range.max,
  )

  if (descriptor.kind === 'joint') {
    const jointId = descriptor.id.replace(/^joint:/, '')
    const executor = options.executorDescriptors.find((item) => item.id === jointId)
    const qposAdr = executor?.qposAddr ?? -1
    const qvelAdr = executor?.qvelAddr ?? -1
    if (qposAdr < 0 || !options.data.qpos) {
      return false
    }
    options.data.qpos[qposAdr] = clampedValue
    if (qvelAdr >= 0 && options.data.qvel) {
      options.data.qvel[qvelAdr] = 0
    }
    return true
  }

  const actuatorId = Number(descriptor.id.replace(/^actuator:/, ''))
  if (!Number.isFinite(actuatorId) || !options.data.ctrl) {
    return false
  }
  options.data.ctrl[actuatorId] = clampedValue
  return true
}
