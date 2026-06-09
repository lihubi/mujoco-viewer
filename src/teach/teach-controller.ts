import type { MujocoExecutorDescriptor } from '../types'
import type {
  MujocoTeachControllerCreateOptions,
  MujocoTeachControllerOptions,
  MujocoTeachJointControl,
  MujocoTeachRuntimeTarget,
  MujocoTeachStepOptions,
} from './types'

type NumericArrayLike = {
  [index: number]: number
  length: number
}

type MutableNumericArrayLike = NumericArrayLike

type MujocoTeachModelReader = {
  opt?: { disableflags?: number }
  nbody?: number
  njnt?: number
  nu?: number
  body_parentid?: NumericArrayLike
  body_jntadr?: NumericArrayLike
  body_jntnum?: NumericArrayLike
  jnt_type?: NumericArrayLike
  jnt_qposadr?: NumericArrayLike
  jnt_dofadr?: NumericArrayLike
  actuator_trntype?: NumericArrayLike
  actuator_trnid?: NumericArrayLike
  actuator_biastype?: NumericArrayLike
  actuator_ctrlrange?: NumericArrayLike
}

type MujocoTeachDataReader = {
  qpos?: MutableNumericArrayLike
  qvel?: MutableNumericArrayLike
  ctrl?: MutableNumericArrayLike
  qfrc_bias?: NumericArrayLike
}

interface FreeJointAnchor {
  qposAddr: number
  qvelAddr: number
  qpos: Float64Array
}

const DEFAULT_OPTIONS: Required<MujocoTeachControllerOptions> = {
  suspendPhysics: true,
  anchorFreeJoint: true,
  torqueDamping: 0.18,
  torqueHoldDamping: 1.6,
  torqueLimit: 25,
  velocityFilterBeta: 0.2,
}

const clampNumber = (value: number, min: number, max: number): number =>
  Math.min(max, Math.max(min, value))

const enumValue = (entry: unknown, fallback: number): number => {
  const maybeValue = entry as { value?: unknown } | null
  return typeof maybeValue?.value === 'number' ? maybeValue.value : fallback
}

const getEnumValues = (target: MujocoTeachRuntimeTarget) => {
  const mujoco = target.mujoco as unknown as {
    mjtTrn?: Record<string, unknown>
    mjtJoint?: Record<string, unknown>
    mjtBias?: Record<string, unknown>
    mjtDisableBit?: Record<string, unknown>
  }
  return {
    trnJoint: enumValue(mujoco.mjtTrn?.mjTRN_JOINT, 0),
    jointFree: enumValue(mujoco.mjtJoint?.mjJNT_FREE, 0),
    jointSlide: enumValue(mujoco.mjtJoint?.mjJNT_SLIDE, 2),
    jointHinge: enumValue(mujoco.mjtJoint?.mjJNT_HINGE, 3),
    biasNone: enumValue(mujoco.mjtBias?.mjBIAS_NONE, 0),
    disableContact: enumValue(mujoco.mjtDisableBit?.mjDSBL_CONTACT, 1 << 4),
    disableGravity: enumValue(mujoco.mjtDisableBit?.mjDSBL_GRAVITY, 1 << 6),
  }
}

type MujocoTeachEnumValues = ReturnType<typeof getEnumValues>

const collectJointIdsOnPathToRoot = (
  model: MujocoTeachModelReader,
  bodyId: number,
): Set<number> => {
  const jointIds = new Set<number>()
  const bodyCount = Number(model.nbody ?? 0)
  const jointCount = Number(model.njnt ?? 0)
  const maxHops = Math.max(bodyCount + 2, 2)
  let currentBodyId = bodyId
  let hops = 0

  while (currentBodyId >= 0 && hops < maxHops) {
    hops += 1
    const jointStart = Number(model.body_jntadr?.[currentBodyId] ?? -1)
    const jointNum = Math.min(Number(model.body_jntnum?.[currentBodyId] ?? 0), 16)
    if (jointStart >= 0 && jointNum > 0) {
      for (let offset = 0; offset < jointNum; offset += 1) {
        const jointId = jointStart + offset
        if (jointId >= 0 && jointId < jointCount) {
          jointIds.add(jointId)
        }
      }
    }
    currentBodyId = Number(model.body_parentid?.[currentBodyId] ?? -1)
  }

  return jointIds
}

const resolveControlRange = (
  model: MujocoTeachModelReader,
  actuatorId: number,
): { min: number; max: number } => {
  const min = Number(model.actuator_ctrlrange?.[actuatorId * 2])
  const max = Number(model.actuator_ctrlrange?.[actuatorId * 2 + 1])
  if (!Number.isFinite(min) || !Number.isFinite(max) || Math.abs(max - min) <= 1e-9) {
    return { min: -Number.MAX_VALUE, max: Number.MAX_VALUE }
  }
  return min <= max ? { min, max } : { min: max, max: min }
}

const getKnownExecutorJointIds = (executors: MujocoExecutorDescriptor[]): Set<number> =>
  new Set(executors.flatMap((executor) => (
    executor.jointNumericId == null ? [] : [executor.jointNumericId]
  )))

export class MujocoTeachController {
  private readonly options: Required<MujocoTeachControllerOptions>
  private readonly enumValues: MujocoTeachEnumValues
  private readonly jointControls: MujocoTeachJointControl[]
  private readonly velocityFiltered: Float64Array
  private readonly qAtDragStart: Float64Array
  private isEnabled = false
  private wasPerturbed = false
  private savedDisableFlags: number | null = null
  private freeJointAnchor: FreeJointAnchor | null = null
  private cachedPerturbBodyId = -1
  private cachedPerturbJointIds: Set<number> | null = null

  constructor(private readonly createOptions: MujocoTeachControllerCreateOptions) {
    this.options = { ...DEFAULT_OPTIONS, ...(createOptions.options ?? {}) }
    this.enumValues = getEnumValues(createOptions.target)
    this.jointControls = this.discoverJointControls()
    this.velocityFiltered = new Float64Array(this.jointControls.length)
    this.qAtDragStart = new Float64Array(this.jointControls.length)
  }

  get enabled(): boolean {
    return this.isEnabled
  }

  getTeachJointControls(): readonly MujocoTeachJointControl[] {
    return this.jointControls
  }

  setEnabled(enabled: boolean): boolean {
    if (this.isEnabled === enabled) {
      return false
    }

    this.isEnabled = enabled
    if (enabled) {
      this.activate()
    } else {
      this.deactivate()
    }
    return true
  }

  beforeStep(stepOptions: MujocoTeachStepOptions): void {
    if (!this.isEnabled) {
      return
    }

    this.applyFreeJointAnchor()
    if (this.jointControls.length === 0) {
      return
    }

    const data = this.createOptions.target.data as unknown as MujocoTeachDataReader
    const model = this.createOptions.target.model as unknown as MujocoTeachModelReader
    const perturbBodyId = stepOptions.perturbBodyId
    const isPerturbed = perturbBodyId != null && perturbBodyId > 0

    if (!isPerturbed) {
      if (this.wasPerturbed) {
        this.alignPositionControlsToCurrentQpos(data)
      }
      this.wasPerturbed = false
      this.cachedPerturbBodyId = -1
      this.cachedPerturbJointIds = null
      return
    }

    if (!this.wasPerturbed) {
      this.captureDragStartQpos(data)
    }

    const localJointIds = this.resolveLocalJointIds(model, perturbBodyId)
    let hasLocalControl = false
    for (let index = 0; index < this.jointControls.length; index += 1) {
      if (localJointIds.has(this.jointControls[index].jointId)) {
        hasLocalControl = true
        break
      }
    }

    for (let index = 0; index < this.jointControls.length; index += 1) {
      const control = this.jointControls[index]
      const isLocal = !hasLocalControl || localJointIds.has(control.jointId)
      if (control.mode === 'position') {
        this.applyPositionTeachControl(data, control, index, isLocal)
      } else {
        this.applyTorqueTeachControl(data, control, index, isLocal)
      }
    }
    this.wasPerturbed = true
  }

  dispose(): void {
    if (this.isEnabled) {
      this.isEnabled = false
      this.deactivate()
    }
  }

  private activate(): void {
    this.wasPerturbed = false
    this.cachedPerturbBodyId = -1
    this.cachedPerturbJointIds = null
    this.velocityFiltered.fill(0)
    this.captureFreeJointAnchor()
    this.applyTeachPhysics()
    this.forward()
  }

  private deactivate(): void {
    const data = this.createOptions.target.data as unknown as MujocoTeachDataReader
    if (this.wasPerturbed) {
      this.alignPositionControlsToCurrentQpos(data)
    }
    this.restoreTeachPhysics()
    this.freeJointAnchor = null
    this.wasPerturbed = false
    this.cachedPerturbBodyId = -1
    this.cachedPerturbJointIds = null
    this.forward()
  }

  private discoverJointControls(): MujocoTeachJointControl[] {
    const target = this.createOptions.target
    const model = target.model as unknown as MujocoTeachModelReader
    const controlByJointId = new Map<number, MujocoTeachJointControl>()
    const knownJointIds = getKnownExecutorJointIds(this.createOptions.executorDescriptors)
    const actuatorCount = Number(model.nu ?? 0)

    for (let actuatorId = 0; actuatorId < actuatorCount; actuatorId += 1) {
      if (Number(model.actuator_trntype?.[actuatorId] ?? -1) !== this.enumValues.trnJoint) {
        continue
      }
      const jointId = Number(model.actuator_trnid?.[actuatorId * 2] ?? -1)
      if (jointId < 0 || controlByJointId.has(jointId)) {
        continue
      }
      if (knownJointIds.size > 0 && !knownJointIds.has(jointId)) {
        continue
      }
      const jointType = Number(model.jnt_type?.[jointId] ?? -1)
      if (jointType !== this.enumValues.jointHinge && jointType !== this.enumValues.jointSlide) {
        continue
      }
      const qposAddr = Number(model.jnt_qposadr?.[jointId] ?? -1)
      const dofAddr = Number(model.jnt_dofadr?.[jointId] ?? -1)
      if (qposAddr < 0 || dofAddr < 0) {
        continue
      }
      const range = resolveControlRange(model, actuatorId)
      controlByJointId.set(jointId, {
        jointId,
        actuatorId,
        qposAddr,
        dofAddr,
        mode: Number(model.actuator_biastype?.[actuatorId] ?? -1) === this.enumValues.biasNone ? 'torque' : 'position',
        ctrlMin: range.min,
        ctrlMax: range.max,
      })
    }

    return Array.from(controlByJointId.values())
  }

  private applyTeachPhysics(): void {
    if (!this.options.suspendPhysics) {
      return
    }
    const model = this.createOptions.target.model as unknown as MujocoTeachModelReader
    if (!model.opt || typeof model.opt.disableflags !== 'number') {
      return
    }
    this.savedDisableFlags = model.opt.disableflags
    model.opt.disableflags |= this.enumValues.disableContact | this.enumValues.disableGravity
  }

  private restoreTeachPhysics(): void {
    const model = this.createOptions.target.model as unknown as MujocoTeachModelReader
    if (this.savedDisableFlags == null || !model.opt) {
      return
    }
    model.opt.disableflags = this.savedDisableFlags
    this.savedDisableFlags = null
  }

  private captureFreeJointAnchor(): void {
    if (!this.options.anchorFreeJoint) {
      return
    }
    const model = this.createOptions.target.model as unknown as MujocoTeachModelReader
    const data = this.createOptions.target.data as unknown as MujocoTeachDataReader
    const jointCount = Number(model.njnt ?? 0)
    for (let jointId = 0; jointId < jointCount; jointId += 1) {
      if (Number(model.jnt_type?.[jointId] ?? -1) !== this.enumValues.jointFree) {
        continue
      }
      const qposAddr = Number(model.jnt_qposadr?.[jointId] ?? -1)
      const qvelAddr = Number(model.jnt_dofadr?.[jointId] ?? -1)
      if (qposAddr < 0 || qvelAddr < 0 || !data.qpos || !data.qvel) {
        return
      }
      const qpos = new Float64Array(7)
      for (let index = 0; index < qpos.length; index += 1) {
        qpos[index] = Number(data.qpos[qposAddr + index] ?? 0)
      }
      this.freeJointAnchor = { qposAddr, qvelAddr, qpos }
      return
    }
  }

  private applyFreeJointAnchor(): void {
    const data = this.createOptions.target.data as unknown as MujocoTeachDataReader
    if (!this.freeJointAnchor || !data.qpos || !data.qvel) {
      return
    }
    for (let index = 0; index < 7; index += 1) {
      data.qpos[this.freeJointAnchor.qposAddr + index] = this.freeJointAnchor.qpos[index]
    }
    for (let index = 0; index < 6; index += 1) {
      data.qvel[this.freeJointAnchor.qvelAddr + index] = 0
    }
  }

  private captureDragStartQpos(data: MujocoTeachDataReader): void {
    for (let index = 0; index < this.jointControls.length; index += 1) {
      const control = this.jointControls[index]
      this.qAtDragStart[index] = Number(data.qpos?.[control.qposAddr] ?? 0)
    }
  }

  private resolveLocalJointIds(model: MujocoTeachModelReader, perturbBodyId: number): Set<number> {
    if (this.cachedPerturbBodyId === perturbBodyId && this.cachedPerturbJointIds) {
      return this.cachedPerturbJointIds
    }
    this.cachedPerturbBodyId = perturbBodyId
    this.cachedPerturbJointIds = collectJointIdsOnPathToRoot(model, perturbBodyId)
    return this.cachedPerturbJointIds
  }

  private applyPositionTeachControl(
    data: MujocoTeachDataReader,
    control: MujocoTeachJointControl,
    index: number,
    isLocal: boolean,
  ): void {
    if (!data.ctrl || !data.qpos) {
      return
    }
    const source = isLocal
      ? Number(data.qpos[control.qposAddr] ?? 0)
      : this.qAtDragStart[index]
    data.ctrl[control.actuatorId] = clampNumber(source, control.ctrlMin, control.ctrlMax)
  }

  private applyTorqueTeachControl(
    data: MujocoTeachDataReader,
    control: MujocoTeachJointControl,
    index: number,
    isLocal: boolean,
  ): void {
    if (!data.ctrl || !data.qvel) {
      return
    }
    const beta = this.options.velocityFilterBeta
    const qvel = Number(data.qvel[control.dofAddr] ?? 0)
    this.velocityFiltered[index] = (1 - beta) * this.velocityFiltered[index] + beta * qvel
    const damping = isLocal ? this.options.torqueDamping : this.options.torqueHoldDamping
    const bias = Number(data.qfrc_bias?.[control.dofAddr] ?? 0)
    const torque = bias - damping * this.velocityFiltered[index]
    data.ctrl[control.actuatorId] = clampNumber(
      clampNumber(torque, -this.options.torqueLimit, this.options.torqueLimit),
      control.ctrlMin,
      control.ctrlMax,
    )
  }

  private alignPositionControlsToCurrentQpos(data: MujocoTeachDataReader): void {
    if (!data.ctrl || !data.qpos) {
      return
    }
    const { ctrl, qpos } = data
    for (let index = 0; index < this.jointControls.length; index += 1) {
      const control = this.jointControls[index]
      if (control.mode !== 'position') {
        continue
      }
      ctrl[control.actuatorId] = clampNumber(
        Number(qpos[control.qposAddr] ?? 0),
        control.ctrlMin,
        control.ctrlMax,
      )
    }
  }

  private forward(): void {
    const target = this.createOptions.target as MujocoTeachRuntimeTarget & {
      mujoco: { mj_forward?: (model: unknown, data: unknown) => void }
    }
    target.mujoco.mj_forward?.(target.model, target.data)
  }
}
