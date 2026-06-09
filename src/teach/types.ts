import type { MujocoData, MujocoExecutorDescriptor, MujocoModel, MujocoModule } from '../types'

export interface MujocoTeachRuntimeTarget {
  mujoco: MujocoModule
  model: MujocoModel
  data: MujocoData
}

export interface MujocoTeachControllerOptions {
  suspendPhysics?: boolean
  anchorFreeJoint?: boolean
  torqueDamping?: number
  torqueHoldDamping?: number
  torqueLimit?: number
  velocityFilterBeta?: number
}

export interface MujocoTeachStepOptions {
  perturbBodyId: number | null
}

export type MujocoTeachJointControlMode = 'position' | 'torque'

export interface MujocoTeachJointControl {
  jointId: number
  actuatorId: number
  qposAddr: number
  dofAddr: number
  mode: MujocoTeachJointControlMode
  ctrlMin: number
  ctrlMax: number
}

export interface MujocoTeachControllerCreateOptions {
  target: MujocoTeachRuntimeTarget
  executorDescriptors: MujocoExecutorDescriptor[]
  options?: MujocoTeachControllerOptions
}
