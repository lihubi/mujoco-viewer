import { describe, expect, it } from 'vitest'
import {
  applyMujocoControlValue,
  buildMujocoRuntimeDescriptorCatalog,
  createMujocoRuntimeSnapshot,
  readMujocoName,
} from '../descriptors'
import type { MujocoModule, MujocoViewerOptionId } from '../types'

const createNameTable = (names: string[]): { names: Uint8Array; offsets: number[] } => {
  const offsets: number[] = []
  const bytes: number[] = []
  names.forEach((name) => {
    offsets.push(bytes.length)
    for (const char of name) {
      bytes.push(char.charCodeAt(0))
    }
    bytes.push(0)
  })
  return {
    names: Uint8Array.from(bytes),
    offsets,
  }
}

const createMujocoStub = (): MujocoModule => ({
  mjtJoint: {
    mjJNT_FREE: { value: 0 },
    mjJNT_BALL: { value: 1 },
    mjJNT_SLIDE: { value: 2 },
    mjJNT_HINGE: { value: 3 },
  },
  mjtTrn: {
    mjTRN_JOINT: { value: 0 },
    mjTRN_JOINTINPARENT: { value: 1 },
    mjTRN_SLIDERCRANK: { value: 2 },
    mjTRN_TENDON: { value: 3 },
    mjTRN_SITE: { value: 4 },
    mjTRN_BODY: { value: 5 },
  },
} as unknown as MujocoModule)

describe('MuJoCo descriptors', () => {
  it('reads null-terminated MuJoCo names', () => {
    const table = createNameTable(['hinge_a', 'act_a'])

    expect(readMujocoName(table, table.offsets[0])).toBe('hinge_a')
    expect(readMujocoName(table, table.offsets[1])).toBe('act_a')
  })

  it('builds joint and actuator descriptors from a MuJoCo model', () => {
    const table = createNameTable(['hinge_a', 'slide_b', 'act_a'])
    const model = {
      names: table.names,
      njnt: 2,
      nu: 1,
      jnt_type: [3, 2],
      jnt_range: [-1, 1, -0.2, 0.4],
      jnt_bodyid: [1, 2],
      jnt_qposadr: [0, 1],
      jnt_dofadr: [0, 1],
      name_jntadr: [table.offsets[0], table.offsets[1]],
      actuator_trnid: [0, -1],
      actuator_ctrlrange: [-0.5, 0.5],
      name_actuatoradr: [table.offsets[2]],
    }

    const catalog = buildMujocoRuntimeDescriptorCatalog(createMujocoStub(), model)

    expect(catalog.executorDescriptors).toHaveLength(2)
    expect(catalog.controlDescriptors.map((item) => item.id)).toEqual([
      'joint:hinge_a',
      'joint:slide_b',
      'actuator:0',
    ])
    expect(catalog.controlDescriptors[0].unitKind).toBe('angle')
    expect(catalog.controlDescriptors[1].unitKind).toBe('distance')
    expect(catalog.controlDescriptors[2].range).toEqual({ min: -0.5, max: 0.5 })
  })

  it('writes clamped joint and actuator values into qpos and ctrl', () => {
    const table = createNameTable(['hinge_a', 'act_a'])
    const model = {
      names: table.names,
      njnt: 1,
      nu: 1,
      jnt_type: [3],
      jnt_range: [-1, 1],
      jnt_bodyid: [1],
      jnt_qposadr: [0],
      jnt_dofadr: [0],
      name_jntadr: [table.offsets[0]],
      actuator_trnid: [0, -1],
      actuator_ctrlrange: [-0.25, 0.25],
      name_actuatoradr: [table.offsets[1]],
    }
    const catalog = buildMujocoRuntimeDescriptorCatalog(createMujocoStub(), model)
    const data = {
      qpos: [0],
      qvel: [0.1],
      ctrl: [0],
    }

    expect(applyMujocoControlValue({
      controlId: 'joint:hinge_a',
      value: 5,
      runState: 'paused',
      controlDescriptors: catalog.controlDescriptors,
      executorDescriptors: catalog.executorDescriptors,
      data,
    })).toBe(true)
    expect(data.qpos[0]).toBe(1)
    expect(data.qvel[0]).toBe(0)

    expect(applyMujocoControlValue({
      controlId: 'actuator:0',
      value: -1,
      runState: 'running',
      controlDescriptors: catalog.controlDescriptors,
      executorDescriptors: catalog.executorDescriptors,
      data,
    })).toBe(true)
    expect(data.ctrl[0]).toBe(-0.25)
  })

  it('creates snapshots with control values and viewer option states', () => {
    const table = createNameTable(['hinge_a'])
    const model = {
      names: table.names,
      njnt: 1,
      nu: 0,
      jnt_type: [3],
      jnt_range: [-1, 1],
      jnt_bodyid: [1],
      jnt_qposadr: [0],
      jnt_dofadr: [0],
      name_jntadr: [table.offsets[0]],
    }
    const catalog = buildMujocoRuntimeDescriptorCatalog(createMujocoStub(), model)
    const viewerOptionState = new Map<MujocoViewerOptionId, boolean>([['joint', true]])

    const snapshot = createMujocoRuntimeSnapshot({
      controlDescriptors: catalog.controlDescriptors,
      executorDescriptors: catalog.executorDescriptors,
      data: { time: 1.5, qpos: [0.25] },
      runState: 'running',
      teachModeEnabled: true,
      viewerOptionState,
      interaction: {
        hoveredBodyId: null,
        draggedBodyId: null,
        selectedBodyId: null,
        perturbBodyId: null,
        activePerturbMode: null,
        selectPoint: null,
      },
    })

    expect(snapshot.timeSeconds).toBe(1.5)
    expect(snapshot.teachModeEnabled).toBe(true)
    expect(snapshot.controlValues['joint:hinge_a']).toBe(0.25)
    expect(snapshot.viewerOptionStates.joint).toBe(true)
  })

  it('does not read MuJoCo bool memory view fields when inferring ranges', () => {
    const table = createNameTable(['hinge_a', 'act_a'])
    const model = {
      names: table.names,
      njnt: 1,
      nu: 1,
      jnt_type: [3],
      jnt_range: [-0.75, 0.75],
      jnt_bodyid: [1],
      jnt_qposadr: [0],
      jnt_dofadr: [0],
      name_jntadr: [table.offsets[0]],
      actuator_trnid: [0, -1],
      actuator_ctrlrange: [-0.25, 0.25],
      name_actuatoradr: [table.offsets[1]],
      get jnt_limited() {
        throw new Error('jnt_limited should not be read')
      },
      get actuator_ctrllimited() {
        throw new Error('actuator_ctrllimited should not be read')
      },
    }

    const catalog = buildMujocoRuntimeDescriptorCatalog(createMujocoStub(), model)

    expect(catalog.controlDescriptors[0].range).toEqual({ min: -0.75, max: 0.75 })
    expect(catalog.controlDescriptors[1].range).toEqual({ min: -0.25, max: 0.25 })
  })

  it('keeps non-joint actuator transmissions as standalone actuator controls', () => {
    const table = createNameTable([
      'hinge_a',
      'spatial_tendon',
      'camera_site',
      'crank_site',
      'payload_body',
      'joint_motor',
      'tendon_motor',
      'site_motor',
      'slider_motor',
      'body_motor',
    ])
    const model = {
      names: table.names,
      njnt: 1,
      nu: 5,
      jnt_type: [3],
      jnt_range: [-1, 1],
      jnt_bodyid: [1],
      jnt_qposadr: [0],
      jnt_dofadr: [0],
      name_jntadr: [table.offsets[0]],
      name_tendonadr: [table.offsets[1]],
      name_siteadr: [table.offsets[2], table.offsets[3]],
      name_bodyadr: [0, table.offsets[4]],
      actuator_trntype: [0, 3, 4, 2, 5],
      actuator_trnid: [
        0, -1,
        0, -1,
        0, -1,
        1, 0,
        1, -1,
      ],
      actuator_ctrlrange: [
        -1, 1,
        0, 2,
        -0.5, 0.5,
        -2, 2,
        -3, 3,
      ],
      name_actuatoradr: [
        table.offsets[5],
        table.offsets[6],
        table.offsets[7],
        table.offsets[8],
        table.offsets[9],
      ],
    }

    const catalog = buildMujocoRuntimeDescriptorCatalog(createMujocoStub(), model)
    const actuators = catalog.controlDescriptors.filter((item) => item.kind === 'actuator')

    expect(actuators.map((item) => item.label)).toEqual([
      'joint_motor',
      'tendon_motor',
      'site_motor',
      'slider_motor',
      'body_motor',
    ])
    expect(actuators.map((item) => item.transmissionType)).toEqual([
      'joint',
      'tendon',
      'site',
      'slidercrank',
      'body',
    ])
    expect(actuators.find((item) => item.label === 'tendon_motor')?.sourceLabel).toBe('tendon:spatial_tendon')
    expect(actuators.find((item) => item.label === 'site_motor')?.sourceLabel).toBe('site:camera_site')
    expect(catalog.executorDescriptors).toHaveLength(1)
  })
})
