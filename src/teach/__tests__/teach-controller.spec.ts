import { describe, expect, it, vi } from 'vitest'
import { MujocoTeachController } from '../teach-controller'
import type { MujocoExecutorDescriptor, MujocoRuntimeTarget } from '../../types'

const createMujocoStub = () => ({
  mjtTrn: { mjTRN_JOINT: { value: 0 } },
  mjtJoint: {
    mjJNT_FREE: { value: 0 },
    mjJNT_SLIDE: { value: 2 },
    mjJNT_HINGE: { value: 3 },
  },
  mjtBias: { mjBIAS_NONE: { value: 0 } },
  mjtDisableBit: {
    mjDSBL_CONTACT: { value: 1 << 4 },
    mjDSBL_GRAVITY: { value: 1 << 6 },
  },
  mj_forward: vi.fn(),
})

const executors: MujocoExecutorDescriptor[] = [
  {
    id: 'hinge',
    name: 'hinge',
    jointType: 'hinge',
    jointTypeLabel: 'Hinge',
    summary: '',
    rangeLabel: '',
    sourceLabel: 'hinge',
    jointNumericId: 1,
    bodyId: 1,
    qposAddr: 7,
    qvelAddr: 6,
    dofAdr: 6,
    actuatorId: 0,
    ctrlAdr: 0,
  },
]

const createTarget = (): MujocoRuntimeTarget => ({
  mujoco: createMujocoStub() as unknown as MujocoRuntimeTarget['mujoco'],
  model: {
    opt: { disableflags: 2 },
    nbody: 2,
    njnt: 2,
    nu: 1,
    body_parentid: [-1, 0],
    body_jntadr: [0, 1],
    body_jntnum: [1, 1],
    jnt_type: [0, 3],
    jnt_qposadr: [0, 7],
    jnt_dofadr: [0, 6],
    actuator_trntype: [0],
    actuator_trnid: [1, -1],
    actuator_biastype: [1],
    actuator_ctrlrange: [-1, 1],
  } as unknown as MujocoRuntimeTarget['model'],
  data: {
    qpos: [0, 0, 0, 1, 0, 0, 0, 0.2],
    qvel: [0, 0, 0, 0, 0, 0, 0.4],
    ctrl: [0],
    qfrc_bias: [0, 0, 0, 0, 0, 0, 0],
  } as unknown as MujocoRuntimeTarget['data'],
})

describe('MujocoTeachController', () => {
  it('discovers joint actuators and aligns position controls during drag and release', () => {
    const target = createTarget()
    const data = target.data as unknown as { qpos: number[]; ctrl: number[] }
    const controller = new MujocoTeachController({ target, executorDescriptors: executors })

    expect(controller.getTeachJointControls()).toHaveLength(1)
    expect(controller.setEnabled(true)).toBe(true)
    data.qpos[7] = 0.65

    controller.beforeStep({ perturbBodyId: 1 })
    expect(data.ctrl[0]).toBeCloseTo(0.65)

    data.qpos[7] = 0.5
    controller.beforeStep({ perturbBodyId: null })
    expect(data.ctrl[0]).toBeCloseTo(0.5)
  })

  it('does not read MuJoCo bool memory view fields when discovering teach controls', () => {
    const target = createTarget()
    Object.defineProperty(target.model, 'actuator_ctrllimited', {
      get() {
        throw new Error('actuator_ctrllimited should not be read')
      },
    })

    const controller = new MujocoTeachController({ target, executorDescriptors: executors })

    expect(controller.getTeachJointControls()[0]).toMatchObject({
      ctrlMin: -1,
      ctrlMax: 1,
    })
  })

  it('suspends physics, anchors the first freejoint, and restores disableflags', () => {
    const target = createTarget()
    const model = target.model as unknown as { opt: { disableflags: number } }
    const data = target.data as unknown as { qpos: number[]; qvel: number[] }
    const controller = new MujocoTeachController({ target, executorDescriptors: executors })

    controller.setEnabled(true)
    expect(model.opt.disableflags).toBe(2 | (1 << 4) | (1 << 6))

    data.qpos[0] = 9
    data.qvel[0] = 9
    controller.beforeStep({ perturbBodyId: null })
    expect(data.qpos[0]).toBe(0)
    expect(data.qvel[0]).toBe(0)

    controller.setEnabled(false)
    expect(model.opt.disableflags).toBe(2)
  })
})
